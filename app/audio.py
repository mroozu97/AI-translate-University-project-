# audio.py
import queue
import threading

import numpy as np
import sounddevice as sd
import deepl
from scipy.signal import butter, lfilter, resample_poly

from ai_controller import AIController, AIControllerConfig, decide_send_to_asr


def start_live_listener(
    whisper_model,
    deepl_client: deepl.DeepLClient,
    whisper_language: str = "pl",
    deepl_source_lang: str = "PL",
    deepl_target_lang: str = "EN-GB",
    input_samplerate: int = 48000,
    whisper_samplerate: int = 16000,
    chunk_ms: int = 30,
    start_threshold: float = 0.018,
    silence_threshold: float = 0.012,
    silence_ms_to_end: int = 650,
    max_phrase_ms: int = 6000,
    highpass_hz: float = 80.0,
    limiter_target: float = 0.9,
    debug_ai: bool = False,
):
    """
    Live listener (bez DeepFilterNet):
    - nagrywanie w 48kHz
    - high-pass + limiter (DSP)
    - autorski AIController: klasyfikacja segmentu (SILENCE/NOISE/SPEECH/COMMAND) + intent detection (STOP/CHANGE_LANG)
    - resampling do 16kHz dla Whisper
    - transkrypcja + tłumaczenie DeepL
    - STOP oraz zmiana języka tłumaczenia komendą głosową
    """

    # ---- Autorski kontroler AI (nasz moduł) ----
    # cfg = AIControllerConfig(samplerate=input_samplerate)
    cfg = AIControllerConfig(
        samplerate=input_samplerate,
        min_segment_ms=120,
        speech_score_threshold=0.48,
        command_rms_factor=1.8,
    )

    controller = AIController(cfg)

    # ---- Język docelowy tłumaczenia (zmieniany komendą) ----
    lang_lock = threading.Lock()
    current_target_lang = {"code": deepl_target_lang}

    # ---- Kolejka fraz + stop flag ----
    phrase_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=5)
    stop_flag = threading.Event()

    chunk_samples = int(input_samplerate * chunk_ms / 1000)
    silence_chunks_to_end = max(1, silence_ms_to_end // chunk_ms)
    max_phrase_chunks = max(1, max_phrase_ms // chunk_ms)

    # ---- DSP ----
    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

    def butter_highpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        norm = cutoff / nyq
        return butter(order, norm, btype="highpass")

    def apply_highpass(x: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
        if cutoff <= 0:
            return x
        b, a = butter_highpass(cutoff, fs)
        return lfilter(b, a, x).astype(np.float32)

    def limiter(x: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        peak = float(np.max(np.abs(x)) + 1e-12)
        if peak > target_peak:
            x *= target_peak / peak
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def enhance_pipeline(x: np.ndarray) -> np.ndarray:
        x = (x - np.mean(x)).astype(np.float32)          # DC offset
        x = apply_highpass(x, input_samplerate, highpass_hz)
        x = limiter(x, limiter_target)
        return x.astype(np.float32)

    def to_whisper_rate(x: np.ndarray) -> np.ndarray:
        return resample_poly(x, whisper_samplerate, input_samplerate).astype(np.float32)

    # ---- Worker: AI gating -> Whisper -> intent -> DeepL ----
    def worker():
        while not stop_flag.is_set():
            try:
                audio_48k = phrase_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # DSP
            audio_48k = enhance_pipeline(audio_48k.astype(np.float32))

            # ✅ Nasz AI: decyzja czy segment wysyłać do ASR
            send, feats, cls = decide_send_to_asr(controller, audio_48k)

            if debug_ai:
                print(
                    f"[AI] cls={cls:<7} score={feats.score:.2f} "
                    f"rms={feats.rms:.4f} zcr={feats.zcr:.3f} "
                    f"cent={feats.centroid_hz:.0f}Hz tilt={feats.tilt_db:.1f}"
                )

            if not send:
                # SILENCE/NOISE -> pomijamy (oszczędzamy Whisper)
                continue

            # Resampling pod Whisper
            audio_16k = to_whisper_rate(audio_48k)

            # delikatne podbicie jeśli za cicho
            peak = float(np.max(np.abs(audio_16k)) + 1e-12)
            if peak < 0.15:
                audio_16k = np.clip(audio_16k * (0.15 / peak), -1.0, 1.0).astype(np.float32)

            # ASR
            segments, _ = whisper_model.transcribe(
                audio_16k,
                language=whisper_language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            texts = [s.text.strip() for s in segments if (s.text or "").strip()]
            if not texts:
                continue

            original = " ".join(texts).strip()

            # ✅ Nasz AI: intent detection na bazie transkrypcji
            intent = controller.detect_intent(original)

            if intent.type == "STOP":
                print("\n☠️ Wykryto frazę kończącą:")
                print(f"👉 \"{original}\"")
                print("⏹️ Zamykanie programu...")
                stop_flag.set()
                return

            if intent.type == "CHANGE_LANG" and intent.payload:
                with lang_lock:
                    current_target_lang["code"] = intent.payload
                print("\n🔁 Zmieniono język tłumaczenia na:", intent.payload)
                print("-" * 40)
                continue  # nie tłumacz samej komendy

            # NORMALNE TŁUMACZENIE
            with lang_lock:
                target_lang_now = current_target_lang["code"]

            print("\n📝 Oryginalny:", original)
            translated = deepl_client.translate_text(
                original,
                source_lang=deepl_source_lang,
                target_lang=target_lang_now,
                formality=deepl.Formality.DEFAULT,
            )
            print(f"🌍 Tłumaczenie ({target_lang_now}):", translated.text)
            print("-" * 40)

    threading.Thread(target=worker, daemon=True).start()

    print("🎧 Start nasłuchu (CTRL+C aby przerwać)")
    print("🗣️ Komendy:")
    print(" - „tłumacz na angielski / niemiecki / francuski ...”")
    print(" - „zmień język na angielski amerykański”")
    print(" - STOP: „Żegnaj, Gulu. Widzimy się w piekle.”")
    if debug_ai:
        print(" - debug_ai=True: wypisuje cechy i klasyfikację AI")
    print("-" * 40)

    # ---- Pętla nagrywania + segmentacja RMS ----
    started = False
    silent_chunks = 0
    frames: list[np.ndarray] = []
    phrase_chunks = 0

    try:
        with sd.InputStream(
            samplerate=input_samplerate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            latency="low",
        ) as stream:
            while not stop_flag.is_set():
                data, _overflowed = stream.read(chunk_samples)
                x = data[:, 0].astype(np.float32)
                level = rms(x)

                if not started:
                    if level >= start_threshold:
                        started = True
                        frames = [x.copy()]
                        silent_chunks = 0
                        phrase_chunks = 1
                else:
                    frames.append(x.copy())
                    phrase_chunks += 1

                    if level < silence_threshold:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    if silent_chunks >= silence_chunks_to_end or phrase_chunks >= max_phrase_chunks:
                        try:
                            phrase_queue.put_nowait(np.concatenate(frames))
                        except queue.Full:
                            pass

                        started = False
                        frames.clear()
                        silent_chunks = 0
                        phrase_chunks = 0

    except KeyboardInterrupt:
        print("\n⏹️ Stop (CTRL+C)")
    finally:
        stop_flag.set()
        print("✅ Zakończono.")

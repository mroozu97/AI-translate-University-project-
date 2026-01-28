import time
import queue
import threading
import numpy as np
import sounddevice as sd
import deepl

# brak adnotacji WhisperModel, żeby nie robić problemów przy imporcie


def start_live_listener(
    whisper_model,
    deepl_client: deepl.DeepLClient,
    whisper_language: str = "pl",
    deepl_source_lang: str = "PL",
    deepl_target_lang: str = "EN-GB",
    samplerate: int = 16000,
    chunk_ms: int = 30,
    start_threshold: float = 0.012,
    silence_threshold: float = 0.010,
    silence_ms_to_end: int = 600,   # krócej = szybciej kończy frazę
    max_phrase_ms: int = 5000,      # maks długość jednej frazy
):
    """
    - słucha mikrofonu w czasie rzeczywistym
    - dzieli na frazy (start mowy -> cisza)
    - transkrybuje + tłumaczy w osobnym wątku
    """

    phrase_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=5)
    stop_flag = threading.Event()

    chunk_samples = int(samplerate * chunk_ms / 1000)
    silence_chunks_to_end = max(1, silence_ms_to_end // chunk_ms)
    max_phrase_chunks = max(1, max_phrase_ms // chunk_ms)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

    def worker():
        while not stop_flag.is_set():
            try:
                audio = phrase_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # faster-whisper lubi float32 w [-1, 1]
            audio = audio.astype(np.float32)
            print(f"[dbg] len={len(audio) / samplerate:.2f}s peak={np.max(np.abs(audio)):.2f}")
            # Szybsze i „czystsze” wyniki: vad_filter usuwa ciszę, zwykle przyspiesza

            segments, info = whisper_model.transcribe(
                audio,
                language=whisper_language,
                vad_filter=True,  # WŁĄCZ (często pomaga w live)
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=1,  # szybciej i stabilniej w live
                temperature=0.0,
                condition_on_previous_text=False,  # wyłącz, bo przy złych frazach psuje
            )

            texts = [s.text.strip() for s in segments if (s.text or "").strip()]
            if not texts:
                continue

            original = " ".join(texts).strip()
            print("\n📝 Oryginalny:", original)

            # Tłumaczymy całość naraz (szybciej i lepiej jakościowo)
            translated = deepl_client.translate_text(
                original,
                source_lang=deepl_source_lang,
                target_lang=deepl_target_lang,
                formality=deepl.Formality.DEFAULT,
            )
            print("🌍 Tłumaczenie:", translated.text)
            print("-" * 40)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    print("🎧 Start nasłuchu (CTRL+C aby przerwać)")

    started = False
    silent_chunks = 0
    frames = []
    phrase_chunks = 0

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", blocksize=chunk_samples) as stream:
            while True:
                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    # jak masz overflow, to zmniejsz samplerate albo chunk_ms zwiększ
                    pass

                x = data[:, 0]  # mono
                level = rms(x)

                if not started:
                    if level >= start_threshold:
                        started = True
                        silent_chunks = 0
                        frames = [x.copy()]
                        phrase_chunks = 1
                else:
                    frames.append(x.copy())
                    phrase_chunks += 1

                    if level < silence_threshold:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    # koniec frazy po ciszy albo po limicie długości
                    if silent_chunks >= silence_chunks_to_end or phrase_chunks >= max_phrase_chunks:
                        audio = np.concatenate(frames, axis=0)

                        # lekka normalizacja (bez kosztów dysku)
                        peak = float(np.max(np.abs(audio)) + 1e-9)
                        if peak < 0.2:
                            audio = np.clip(audio * (0.2 / peak), -1.0, 1.0)

                        # wrzuć do kolejki (jak pełna — pomijamy, żeby nie lagować)
                        try:
                            phrase_queue.put_nowait(audio)
                        except queue.Full:
                            pass

                        started = False
                        frames = []
                        silent_chunks = 0
                        phrase_chunks = 0

    except KeyboardInterrupt:
        print("\n⏹️ Stop")
    finally:
        stop_flag.set()

import queue
import threading
import re

import numpy as np
import sounddevice as sd
import deepl

from scipy.signal import butter, lfilter, resample_poly

# --- AI denoiser (DeepFilterNet) ---
try:
    from df.enhance import enhance, init_df
    _DF_MODEL, _DF_STATE, _ = init_df()
    HAS_DF = True
except Exception:
    HAS_DF = False


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
    use_ai_denoise: bool = True,
    highpass_hz: float = 80.0,
    limiter_target: float = 0.9,
):
    """
    Live listener:
    - nagrywanie w 48kHz
    - (opcjonalnie) AI denoise (DeepFilterNet) per fraza
    - high-pass + limiter
    - resampling do 16kHz dla Whisper
    - transkrypcja + tłumaczenie DeepL
    - STOP jeśli usłyszy zadaną frazę
    - ZMIANA JĘZYKA tłumaczenia komendą głosową w trakcie działania
    """

    # =========================
    # Komendy głosowe
    # =========================

    # Fraza kończąca (znormalizowana)
    STOP_PHRASES = [
        "zegnaj gulu widzimy sie w piekle",
        "żegnaj gulu widzimy się w piekle",
        "żegnaj gólu widzimy się w piekle",
    ]

    # Mapowanie nazw języków (po polsku i po angielsku) -> kody DeepL target_lang

    LANG_ALIASES = {
        # English
        "angielski": "EN-GB",
        "english": "EN-GB",
        "angielski brytyjski": "EN-GB",
        "english uk": "EN-GB",
        "angielski amerykanski": "EN-US",
        "angielski amerykański": "EN-US",
        "english us": "EN-US",
        "american english": "EN-US",

        # German
        "niemiecki": "DE",
        "german": "DE",

        # French
        "francuski": "FR",
        "french": "FR",

        # Spanish
        "hiszpanski": "ES",
        "hiszpański": "ES",
        "spanish": "ES",

        # Italian
        "wloski": "IT",
        "włoski": "IT",
        "italian": "IT",

        # Dutch
        "holenderski": "NL",
        "niderlandzki": "NL",
        "dutch": "NL",

        # Polish
        "polski": "PL",
        "polish": "PL",

        # Portuguese
        "portugalski": "PT-PT",     # możesz woleć PT-PT
        "portuguese": "PT-PT",
        "portugalski brazylijski": "PT-BR",
        "brazylijski portugalski": "PT-BR",
        "portuguese brazil": "PT-BR",

        # Japanese
        "japonski": "JA",
        "japoński": "JA",
        "japanese": "JA",

        # Chinese (simplified)
        "chinski": "ZH",
        "chiński": "ZH",
        "chinese": "ZH",

        # Russian
        "rosyjski": "RU",
        "russian": "RU",

        # Ukrainian
        "ukrainski": "UK",
        "ukraiński": "UK",
        "ukrainian": "UK",
    }

    # Wzorce komend zmiany języka
    # Przykłady, które zadziałają:
    # - "zmień język na angielski"
    # - "zmien jezyk na niemiecki"
    # - "tłumacz na francuski"
    # - "tlumacz na spanish"
    # - "ustaw język tłumaczenia na włoski"
    CHANGE_LANG_PATTERNS = [
        r"\bzmie[nń]\s+jezyk\s+na\s+(.+)$",
        r"\bzmie[nń]\s+język\s+na\s+(.+)$",
        r"\bt[łl]umacz\s+na\s+(.+)$",
        r"\bustaw\s+jezyk\s+(?:t[łl]umaczenia\s+)?na\s+(.+)$",
        r"\bustaw\s+język\s+(?:t[łl]umaczenia\s+)?na\s+(.+)$",
        r"\btranslate\s+to\s+(.+)$",
    ]

    # Aktualny język docelowy (może się zmieniać w trakcie działania)
    lang_lock = threading.Lock()
    current_target_lang = {"code": deepl_target_lang}  # trzymamy w dict, żeby łatwo modyfikować z wnętrza funkcji

    def normalize_text(txt: str) -> str:
        txt = txt.lower()
        txt = re.sub(r"[^\w\sąćęłńóśżź]", "", txt)   # usuń interpunkcję, zostaw polskie znaki
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def try_parse_lang_command(normalized_text: str) -> str | None:
        """
        Zwraca kod języka DeepL (np. 'DE', 'EN-GB') jeśli wykryje komendę zmiany języka,
        w przeciwnym razie None.
        """
        for pat in CHANGE_LANG_PATTERNS:
            m = re.search(pat, normalized_text)
            if not m:
                continue

            raw_lang = m.group(1).strip()
            # czasem whisper dopisze końcówki typu "proszę", "teraz" — obetnij na końcu
            raw_lang = re.sub(r"\b(prosze|proszę|teraz|dziekuje|dziękuję)\b$", "", raw_lang).strip()

            # dopasowanie aliasów
            if raw_lang in LANG_ALIASES:
                return LANG_ALIASES[raw_lang]

            # spróbuj dopasować po "pierwszych słowach"
            # np. "angielski brytyjski" itp.
            # albo gdy whisper rozbije: "angielski brytyjski prosze"
            for k, v in LANG_ALIASES.items():
                if raw_lang.startswith(k):
                    return v

            return None

        return None

    # =========================
    # Audio pipeline
    # =========================

    phrase_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=5)
    stop_flag = threading.Event()

    chunk_samples = int(input_samplerate * chunk_ms / 1000)
    silence_chunks_to_end = max(1, silence_ms_to_end // chunk_ms)
    max_phrase_chunks = max(1, max_phrase_ms // chunk_ms)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

    def butter_highpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        norm = cutoff / nyq
        return butter(order, norm, btype="highpass")

    def apply_highpass(x: np.ndarray, fs: int) -> np.ndarray:
        if highpass_hz <= 0:
            return x
        b, a = butter_highpass(highpass_hz, fs)
        return lfilter(b, a, x).astype(np.float32)

    def limiter(x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x)) + 1e-12)
        if peak > limiter_target:
            x *= limiter_target / peak
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def ai_denoise(x: np.ndarray) -> np.ndarray:
        if not (use_ai_denoise and HAS_DF):
            return x
        try:
            y, _ = enhance(_DF_MODEL, _DF_STATE, x)
            return y.astype(np.float32)
        except Exception:
            return x

    def enhance_pipeline(x: np.ndarray) -> np.ndarray:
        x = (x - np.mean(x)).astype(np.float32)  # DC offset
        x = apply_highpass(x, input_samplerate)  # HPF
        x = ai_denoise(x)                        # AI denoise
        x = limiter(x)                           # limiter
        return x.astype(np.float32)

    def to_whisper_rate(x: np.ndarray) -> np.ndarray:
        return resample_poly(x, whisper_samplerate, input_samplerate).astype(np.float32)

    def worker():
        while not stop_flag.is_set():
            try:
                audio_48k = phrase_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_48k = enhance_pipeline(audio_48k)
            audio_16k = to_whisper_rate(audio_48k)

            # delikatne podbicie jeśli za cicho
            peak = float(np.max(np.abs(audio_16k)) + 1e-12)
            if peak < 0.15:
                audio_16k = np.clip(audio_16k * (0.15 / peak), -1.0, 1.0).astype(np.float32)

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
            normalized = normalize_text(original)

            # ✅ 1) STOP
            for stop_phrase in STOP_PHRASES:
                if stop_phrase in normalized:
                    print("\n☠️ Wykryto frazę kończącą:")
                    print(f"👉 \"{original}\"")
                    print("⏹️ Zamykanie programu...")
                    stop_flag.set()
                    return

            # ✅ 2) ZMIANA JĘZYKA
            new_lang = try_parse_lang_command(normalized)
            if new_lang is not None:
                with lang_lock:
                    current_target_lang["code"] = new_lang
                print("\n🔁 Zmieniono język tłumaczenia na:", new_lang)
                print("-" * 40)
                continue  # nie tłumacz samej komendy

            # ✅ 3) NORMALNE TŁUMACZENIE
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
    # if use_ai_denoise and not HAS_DF:
    #     print("⚠️ DeepFilterNet niedostępny (działa bez AI).")

    print("🗣️ Komendy:")
    print(" - „tłumacz na angielski / niemiecki / francuski ...”")
    print(" - „zmień język na angielski amerykański”")
    print(" - STOP: „Żegnaj, Gulu. Widzimy się w piekle.”")
    print("-" * 40)

    started = False
    silent_chunks = 0
    frames = []
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


# if __name__ == "__main__":
#     from faster_whisper import WhisperModel
#     whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
#     deepl_client = deepl.DeepLClient("YOUR_DEEPL_API_KEY")
#     start_live_listener(whisper_model, deepl_client)

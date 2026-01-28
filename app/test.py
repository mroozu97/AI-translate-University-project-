import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from scipy.io import wavfile
from faster_whisper import WhisperModel


SR = 16000
DUR = 6

def record_wav(path="test.wav", input_device=9):  # <- zmień tu 35 na 27 jeśli trzeba
    dev = sd.query_devices(input_device)
    if dev["max_input_channels"] <= 0:
        raise ValueError(f"Device {input_device} nie jest wejściem audio: {dev['name']}")

    sd.default.device = (input_device, None)   # ustaw PRZED nagrywaniem
    print(f"🎤 Mów teraz ({DUR}s)... [device={input_device}: {dev['name']}]")

    audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()

    x = audio[:, 0]
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x*x)) + 1e-12)
    print(f"[dbg] peak={peak:.3f} rms={rms:.4f}")

    wav_int16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    wav_write(path, SR, wav_int16)
    print("💾 zapisano:", path)



def trim_silence_wav(in_path: str, out_path: str, threshold: float = 0.02, pad_ms: int = 250):
    """
    Przytnij ciszę na początku i końcu na podstawie RMS.
    threshold: próg RMS (0..1). Jak utnie za mocno, zmniejsz np. do 0.015.
    pad_ms: ile ms zostawić bufora przed/po mowie.
    """
    sr, data = wavfile.read(in_path)

    # data może być int16 -> zamiana na float [-1,1]
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    else:
        x = data.astype(np.float32)

    # RMS w oknie 20ms
    frame = max(1, int(0.02 * sr))
    rms = np.sqrt(np.convolve(x * x, np.ones(frame) / frame, mode="same"))

    idx = np.where(rms > threshold)[0]
    if len(idx) == 0:
        # nic nie wykryto - skopiuj bez zmian
        wavfile.write(out_path, sr, data)
        print("✂️ trim: nie wykryto mowy (zapis bez zmian)")
        return

    pad = int(pad_ms / 1000 * sr)
    start = max(0, int(idx[0] - pad))
    end = min(len(x), int(idx[-1] + pad))

    trimmed = x[start:end]
    trimmed_int16 = (np.clip(trimmed, -1, 1) * 32767).astype(np.int16)
    wavfile.write(out_path, sr, trimmed_int16)

    print(f"✂️ trim: {in_path} -> {out_path}  (od {start/sr:.2f}s do {end/sr:.2f}s, len={len(trimmed)/sr:.2f}s)")


def transcribe(path="test.wav", trimmed_path="test_trim.wav"):
    # 1) przytnij ciszę
    trim_silence_wav(path, trimmed_path, threshold=0.02, pad_ms=250)

    # 2) whisper
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    # Ustawienia bardziej odporne na “wydaje się, że...” / powtórki
    kwargs = dict(
        language="pl",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        vad_filter=False,  # tu wyłączamy, bo sami przycięliśmy ciszę
    )

    # Te progi nie zawsze są dostępne w każdej wersji faster-whisper,
    # więc dodajemy je ostrożnie (jeśli nie zadziała, po prostu je Osborne usunąć).
    try:
        segments, info = model.transcribe(
            trimmed_path,
            **kwargs,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    except TypeError:
        # jeśli Twoja wersja nie wspiera tych argumentów
        segments, info = model.transcribe(trimmed_path, **kwargs)

    text = " ".join([s.text.strip() for s in segments if (s.text or "").strip()])
    print("📝:", text)


if __name__ == "__main__":
    record_wav("test.wav")
    transcribe("test.wav", "test_trim.wav")

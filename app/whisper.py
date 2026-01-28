from faster_whisper import WhisperModel

def transcribe_whisper(model: WhisperModel, audio_path: str, language: str = "pl") -> list[str]:
    """Zwraca listę segmentów tekstu z Whispera."""
    segments, info = model.transcribe(audio_path, language=language)
    texts = [s.text.strip() for s in segments if (s.text or "").strip()]
    return texts
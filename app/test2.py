import time
import queue
import threading
import speech_recognition as sr
import deepl

DEEPL_AUTH_KEY = "8943ef91-8994-48c9-84e0-d0b6e5d29ccf:fx" # replace with your key
deepl_client = deepl.DeepLClient(DEEPL_AUTH_KEY)

q: queue.Queue[str] = queue.Queue(maxsize=50)

# --- agregacja tekstu ---
buffer_lock = threading.Lock()
text_buffer = []
last_piece_time = 0.0
FLUSH_AFTER = 1.5  # ile sekund ciszy ma oznaczać "koniec zdania"

def flush_loop():
    global text_buffer, last_piece_time
    while True:
        time.sleep(0.1)
        with buffer_lock:
            if text_buffer and (time.time() - last_piece_time) > FLUSH_AFTER:
                full = " ".join(text_buffer).strip()
                text_buffer = []
                if full:
                    try:
                        q.put_nowait(full)
                    except queue.Full:
                        pass

def translator_worker():
    while True:
        text = q.get()
        if text is None:
            break
        try:
            result = deepl_client.translate_text(text, source_lang="PL", target_lang="EN-GB")
            print(f"\n📝 PL: {text}\n🌍 EN: {result.text}\n")
        except Exception as e:
            print("❌ Błąd tłumaczenia:", e)
        q.task_done()

def on_audio(recognizer: sr.Recognizer, audio: sr.AudioData):
    global last_piece_time
    try:
        piece = recognizer.recognize_google(audio, language="pl-PL").strip()
        if not piece:
            return
        print(f"(fragment) {piece}")
        with buffer_lock:
            text_buffer.append(piece)
            last_piece_time = time.time()
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("❌ Błąd STT:", e)

def main():
    r = sr.Recognizer()
    r.pause_threshold = 1.2
    r.non_speaking_duration = 0.4

    mic = sr.Microphone()
    with mic as source:
        print("🔧 Kalibracja (1s ciszy)...")
        r.adjust_for_ambient_noise(source, duration=1.0)

    threading.Thread(target=translator_worker, daemon=True).start()
    threading.Thread(target=flush_loop, daemon=True).start()

    print("🎧 Mów normalnie. Tłumaczę po dłuższej pauzie. CTRL+C aby zakończyć.")
    stop_listening = r.listen_in_background(mic, on_audio, phrase_time_limit=12)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        q.put(None)

if __name__ == "__main__":
    main()

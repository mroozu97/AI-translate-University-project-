import deepl
from faster_whisper import WhisperModel
from audio import start_live_listener

with open("token.txt", "r", encoding="utf-8") as token_file:
    token = token_file.read()

DEEPL_AUTH_KEY = token

def main():
    deepl_client = deepl.DeepLClient(DEEPL_AUTH_KEY)

    # "base" jest OK, ale jak ma być szybciej: "small" bywa w praktyce podobny,
    # a "tiny" jest najszybszy kosztem jakości.
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    start_live_listener(
        whisper_model=whisper_model,
        deepl_client=deepl_client,
        whisper_language="pl",
        deepl_source_lang="PL",
        deepl_target_lang="EN-GB",
        silence_ms_to_end=1400,  # skraca „czekanie na ciszę”
        max_phrase_ms=12000,
        chunk_ms=50,            # krótsze frazy -> szybciej reaguje
    )

if __name__ == "__main__":
    main()

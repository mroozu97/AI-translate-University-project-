import deepl_API
from faster_whisper import WhisperModel

# Deepl API
auth_key = "8943ef91-8994-48c9-84e0-d0b6e5d29ccf:fx" # replace with your key
deepl_client = deepl.DeepLClient(auth_key)

samples_EN = ["I am an example sentence", "I am another sentence"]
samples_PL = ["Moja sąsiadka codziennie tańczy", "Wyobraźnia jest niesamowita"]
samples_DE = ["Krankenwagen", "Adolf ist in Deutschland ein gebräuchlicher Name"]

result = deepl_client.translate_text(
        samples_EN,
        source_lang="EN",
        target_lang="PL",
        formality=deepl.Formality.DEFAULT,
        tag_handling=None)

#print(result)
# for res in result:
#     print(res)

# Whisper
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("Proba.mp3", language="pl")

print("Tekst oryginalny:")

sample_from_mp = []
for s in segments:
    sample_from_mp.append(s.text)
print(sample_from_mp)

result2 = deepl_client.translate_text(
        sample_from_mp,
        source_lang="PL",
        target_lang="EN-GB",
        formality=deepl.Formality.DEFAULT,
        tag_handling=None)
print("Tekst po tłumaczeniu:")

for res in result2:
    print(res)

import deepl

def translate_deepl(client: deepl.DeepLClient, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
    """Tłumaczy listę stringów DeepL i zwraca listę przetłumaczonych stringów."""
    if not texts:
        return []

    results = client.translate_text(
        texts,
        source_lang=source_lang,
        target_lang=target_lang,
        formality=deepl.Formality.DEFAULT,
        tag_handling=None,
    )
    # DeepL zwraca listę obiektów TextResult
    return [r.text for r in results]
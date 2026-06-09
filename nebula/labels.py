"""Closed label set and prompt for Centinela financial sentiment classification."""

LABELS: list[str] = ["positivo", "neutral", "negativo"]

# Maps the source dataset's English answers to our Spanish-first labels.
EN_TO_ES: dict[str, str] = {
    "positive": "positivo",
    "neutral": "neutral",
    "negative": "negativo",
}

SYSTEM_PROMPT_ES: str = (
    "Sos un experto en análisis de sentimiento financiero. "
    "Analizá el sentimiento del texto financiero dado y respondé únicamente "
    "con una sola palabra: positivo, neutral o negativo."
)

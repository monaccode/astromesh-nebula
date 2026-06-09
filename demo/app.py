"""Gradio demo for Centinela-4B financial sentiment (loads the published GGUF on CPU)."""

import os

from nebula.labels import LABELS, SYSTEM_PROMPT_ES
from nebula.validation import constrain_label

_MODEL_REPO = os.environ.get("CENTINELA_GGUF_REPO", "astromesh/Centinela-Qwen3-4B")
_GGUF_FILE = os.environ.get("CENTINELA_GGUF_FILE", "gguf/centinela-4b.Q4_K_M.gguf")

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama.from_pretrained(repo_id=_MODEL_REPO, filename=_GGUF_FILE, n_ctx=1024)
    return _llm


def _predict(text: str) -> str:
    out = _get_llm().create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ES},
            {"role": "user", "content": text},
        ],
        max_tokens=8,
        temperature=0.0,
    )
    return out["choices"][0]["message"]["content"]


def classify_text(text: str) -> str:
    label = constrain_label(_predict(text), LABELS)
    return label if label is not None else "no clasificable"


def build_ui():
    import gradio as gr

    examples = [
        "Las acciones de la empresa subieron 12% tras superar las expectativas de ganancias.",
        "El banco central mantuvo las tasas sin cambios, en línea con lo previsto.",
        "La compañía reportó pérdidas récord y recortará 5.000 empleos.",
    ]
    return gr.Interface(
        fn=classify_text,
        inputs=gr.Textbox(label="Texto financiero (español)"),
        outputs=gr.Label(label="Sentimiento"),
        examples=examples,
        title="Centinela-4B · Sentimiento financiero",
        description="Familia Centinela · Astromesh Nebula. Salida restringida a positivo/neutral/negativo.",
    )


if __name__ == "__main__":
    build_ui().launch()

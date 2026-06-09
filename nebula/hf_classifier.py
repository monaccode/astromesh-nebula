"""A Classifier backed by a transformers chat model (used at eval/integration time)."""

from nebula.labels import SYSTEM_PROMPT_ES


class HFChatClassifier:
    def __init__(self, model_id: str, max_new_tokens: int = 8):
        from transformers import pipeline

        self._pipe = pipeline("text-generation", model=model_id, device_map="auto")
        self._max_new_tokens = max_new_tokens

    def predict(self, text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ES},
            {"role": "user", "content": text},
        ]
        out = self._pipe(
            messages, max_new_tokens=self._max_new_tokens, do_sample=False
        )
        generated = out[0]["generated_text"]
        # transformers returns the full conversation; take the last assistant turn
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)

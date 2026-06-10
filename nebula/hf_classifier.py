"""A Classifier backed by a transformers chat model (used at eval/integration time)."""

from nebula.labels import SYSTEM_PROMPT_ES


class HFChatClassifier:
    def __init__(self, model_id: str, max_new_tokens: int = 8):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs: dict = {"device_map": "auto"}
        if torch.cuda.is_available():
            # 4-bit NF4 keeps both base and tuned 4B models inside a 12GB GPU and
            # matches the Q4 GGUF that ships; fp16 compute (Turing has no bf16).
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
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

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "unsloth",
#   "trl>=0.9",
#   "transformers>=4.44",
#   "datasets>=2.19",
#   "pyyaml>=6.0",
#   "huggingface_hub>=0.24",
# ]
# ///
"""QLoRA fine-tune Qwen3-4B on the Centinela Spanish sentiment dataset.

Run locally (GPU) or on HF Jobs:
    hf jobs uv run --flavor a10g-large scripts/02_train.py -- \
        --config configs/train-4b.yaml --train data/centinela/train.jsonl \
        --out export/adapter
"""

import argparse
import json

import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train-4b.yaml")
    ap.add_argument("--train", default="data/centinela/train.jsonl")
    ap.add_argument("--out", default="export/adapter")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap optimizer steps (>0) for a cheap smoke test; -1 = full run.",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Turing (e.g. RTX 2060, sm_75) has no bf16 — fall back to fp16. Ampere+ uses bf16.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )

    # Resolve the real EOS from the model config BEFORE get_peft_model — Unsloth Zoo
    # rewrites tokenizer.eos_token to a '<EOS_TOKEN>' sentinel during patching, which
    # TRL 0.24 then rejects as out-of-vocab. (Qwen3 EOS is '<|im_end|>', id 151645.)
    eos_id = model.config.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0]
    real_eos = tokenizer.convert_ids_to_tokens(eos_id)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
    )
    tokenizer.eos_token = real_eos  # undo the sentinel injected by get_peft_model

    # Conversational dataset: TRL applies the chat template natively from the
    # "messages" column (no manual formatting_func — that API is brittle across versions).
    dataset = load_dataset("json", data_files=args.train, split="train")

    sft_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=cfg["train"]["epochs"],
        max_steps=args.max_steps,
        learning_rate=cfg["train"]["learning_rate"],
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        seed=cfg["train"]["seed"],
        max_length=cfg["max_seq_length"],
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        report_to="none",
    )
    # Unsloth patches SFTConfig to inject a '<EOS_TOKEN>' sentinel that TRL 0.24 rejects
    # as out-of-vocab. Force the real EOS on the constructed config (the kwarg gets
    # clobbered by the patch, so set the attribute after construction).
    sft_args.eos_token = real_eos
    print(f"using eos_token={real_eos!r}, pad_token={tokenizer.pad_token!r}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"adapter": args.out, "base_model": cfg["base_model"]}))


if __name__ == "__main__":
    main()

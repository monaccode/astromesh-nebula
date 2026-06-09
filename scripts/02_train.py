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

import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train-4b.yaml")
    ap.add_argument("--train", default="data/centinela/train.jsonl")
    ap.add_argument("--out", default="export/adapter")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
    )

    dataset = load_dataset("json", data_files=args.train, split="train")

    def formatting(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting,
        args=SFTConfig(
            output_dir="checkpoints",
            num_train_epochs=cfg["train"]["epochs"],
            learning_rate=cfg["train"]["learning_rate"],
            per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
            gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
            seed=cfg["train"]["seed"],
            max_seq_length=cfg["max_seq_length"],
            logging_steps=10,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"adapter": args.out, "base_model": cfg["base_model"]}))


if __name__ == "__main__":
    main()

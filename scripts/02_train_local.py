"""Local QLoRA fine-tune of Qwen3-4B for Centinela (no Unsloth).

Standard transformers + PEFT + bitsandbytes + TRL path, used when HF Jobs (cloud GPU)
is unavailable (no paid plan). Runs on a local NVIDIA GPU via WSL2. Unsloth is avoided
because its 2026.6.x build globally patches TRL and is incompatible with TRL 0.24
(injects an out-of-vocab '<EOS_TOKEN>' sentinel into SFTConfig).

    python scripts/02_train_local.py --config configs/train-4b.yaml \
        --train data/centinela/train.jsonl --out export/adapter

Produces a standard PEFT LoRA adapter at --out, consumed unchanged by 03_merge.py.
"""

import argparse
import json

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


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
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=-1,
        help="Subsample the (balanced) train set to N examples; -1 = all.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=-1,
        help="Override num_train_epochs; -1 = use the config value.",
    )
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Turing (RTX 2060, sm_75) has no bf16 — compute/train in fp16. Ampere+ uses bf16.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        # Keep pad distinct from EOS so EOS isn't masked out of the labels.
        tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation="sdpa",  # no flash-attn on Turing
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Conversational dataset: TRL applies the chat template natively from "messages".
    dataset = load_dataset("json", data_files=args.train, split="train")
    if args.max_train_samples > 0 and args.max_train_samples < len(dataset):
        # Shuffle (seeded) then take N; the built train set is label-balanced, so a
        # random subset stays ~balanced across the 3 classes.
        dataset = dataset.shuffle(seed=cfg["train"]["seed"]).select(range(args.max_train_samples))
    print(f"training on {len(dataset)} examples")

    epochs = args.epochs if args.epochs > 0 else cfg["train"]["epochs"]

    sft_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=cfg["train"]["learning_rate"],
        per_device_train_batch_size=cfg["train"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["train"]["seed"],
        max_length=cfg["max_seq_length"],
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=125,
        save_total_limit=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_args,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"adapter": args.out, "base_model": cfg["base_model"]}))


if __name__ == "__main__":
    main()

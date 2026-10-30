# =============================================================================
# finetune.py  -  FINE-TUNE BART ON XSUM  (paper §3.3 + §5.3)
# Optional, run separately (GPU strongly recommended):
#     python finetune.py --model facebook/bart-base --train_samples 5000 --epochs 1
# The saved folder can then be typed into the app's sidebar "Model" box.
# =============================================================================

import argparse

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data import load_xsum


def main():
    # -------------------------------------------------------------------------
    # STAGE 1: ARGUMENTS
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/bart-base")  # start from PRE-TRAINED BART
    ap.add_argument("--output_dir", default="bart-xsum-finetuned")
    ap.add_argument("--train_samples", type=int, default=5000)  # XSum train has 204k
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_source", type=int, default=512)
    ap.add_argument("--max_target", type=int, default=64)
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # STAGE 2: LOAD DATA + PRE-TRAINED MODEL
    # -------------------------------------------------------------------------
    train = load_xsum("train").shuffle(seed=42).select(range(args.train_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # -------------------------------------------------------------------------
    # STAGE 3: TOKENIZE
    # Standard seq2seq setup (paper §3.3): article -> ENCODER input,
    # one-sentence summary -> DECODER target (labels).
    # -------------------------------------------------------------------------
    def preprocess(batch):
        enc = tokenizer(batch["document"], max_length=args.max_source, truncation=True)
        enc["labels"] = tokenizer(
            text_target=batch["summary"], max_length=args.max_target, truncation=True
        )["input_ids"]
        return enc

    train = train.map(preprocess, batched=True, remove_columns=train.column_names)

    # -------------------------------------------------------------------------
    # STAGE 4: TRAINING CONFIG
    # label_smoothing_factor=0.1  -> paper §5.3 "label smoothed cross entropy,
    # smoothing parameter 0.1". Generation uses beam 5 as in the paper.
    # -------------------------------------------------------------------------
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        fp16=torch.cuda.is_available(),
        generation_num_beams=5,
        generation_max_length=62,
        logging_steps=25,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),  # pads labels with -100
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: TRAIN + SAVE
    # -------------------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to ./{args.output_dir}  -> paste this path into the app's Model box.")


if __name__ == "__main__":
    main()
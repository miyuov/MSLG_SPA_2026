import os
import torch
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    TrainerCallback
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class ProgressPrinter(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"   > Época {int(state.epoch)} completada.")

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "Helsinki-NLP/opus-mt-es-en"

def train_cv():
    print(f"--- MSLG-SPA 2026: Training Start ---")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    dataset = load_dataset('csv', data_files={'train': 'data/processed/train_cleaned.csv'})['train']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=hf_token)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n>>> Fold {fold+1}/5")
        
        train_sub = dataset.select(train_idx)
        val_sub = dataset.select(val_idx)

        def preprocess(examples):
            return tokenizer(
                [str(x) for x in examples["MSLG"]], 
                text_target=[str(x) for x in examples["SPA"]], 
                max_length=128, 
                truncation=True, 
                padding="max_length"
            )

        tokenized_train = train_sub.map(preprocess, batched=True, remove_columns=dataset.column_names)
        tokenized_val = val_sub.map(preprocess, batched=True, remove_columns=dataset.column_names)

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, token=hf_token).to(device)

        args = Seq2SeqTrainingArguments(
            output_dir=f"./models/mslg2spa_results/checkpoints_fold_{fold}",
            eval_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            num_train_epochs=15, 
            predict_with_generate=True,
            fp16=True,
            save_total_limit=1,
            logging_steps=20,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer, 
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            callbacks=[ProgressPrinter()]
        )

        trainer.train()
        trainer.save_model(f"./models/mslg2spa_results/final_model_fold_{fold}")
        print(f"Fold {fold+1} guardado.")

if __name__ == "__main__":
    train_cv()
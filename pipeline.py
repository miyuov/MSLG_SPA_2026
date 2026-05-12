"""
MSLG-SPA 2026 — IberLEF
Traducción bidireccional: Glosas LSM <-> Español
Pipeline completo: preparación, entrenamiento, inferencia, evaluación y submission.

Uso:
    uv run python pipeline.py --task all
    uv run python pipeline.py --task train
    uv run python pipeline.py --task inference
    uv run python pipeline.py --task evaluate
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR      = Path(".")
DATA_RAW      = BASE_DIR / "data" / "raw"
DATA_PROC     = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
SUBMISSIONS   = BASE_DIR / "submissions"

CHECKPOINT    = "google/mt5-small"

N_FOLDS       = 4
EPOCHS        = 65
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
LEARNING_RATE = 5e-4
MAX_LEN       = 128
NUM_BEAMS     = 4

PREFIX_MSLG2SPA = "traduce glosa LSM al español: "
PREFIX_SPA2MSLG = "traduce español a glosa LSM: "

TEAM = "Señas del Desierto"
RUN  = "run1_mt5small"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # usar solo una GPU

# =============================================================================
# UTILIDADES
# =============================================================================

def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def check_hardware():
    banner("Verificación de hardware")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Dispositivo: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:        {name}")
        print(f"VRAM:       {vram:.1f} GB")
        if vram < 4:
            print("Nota: VRAM limitada — considera reducir BATCH_SIZE a 2.")
        elif vram < 6:
            print("Nota: VRAM aceptable — los parámetros actuales funcionan bien.")
        else:
            print("Nota: VRAM suficiente — puedes subir BATCH_SIZE a 8 si quieres.")
    else:
        print("Nota: Sin GPU, el entrenamiento será lento.")

def make_dirs():
    for d in [
        DATA_PROC,
        MODELS_DIR / "mslg2spa_final",
        MODELS_DIR / "spa2mslg_final",
        SUBMISSIONS,
    ]:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. PREPARACIÓN DE DATOS
# =============================================================================

def load_train_data():
    path = DATA_RAW / "MSLG_SPA_train.txt"
    df = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["MSLG", "SPA"])
    df["MSLG"] = df["MSLG"].astype(str).str.strip()
    df["SPA"]  = df["SPA"].astype(str).str.strip()
    df = df[(df["MSLG"].str.len() > 1) & (df["SPA"].str.len() > 1)]
    df = df.reset_index(drop=True)
    df.to_csv(DATA_PROC / "train_cleaned.csv", index=False)
    print(f"Datos de entrenamiento: {len(df)} pares")
    return df

def load_test_data(filename, col):
    path = DATA_RAW / filename
    df = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(str).str.strip()
    return df

# =============================================================================
# 2. TOKENIZACIÓN
# =============================================================================

def make_preprocess_fn(tokenizer, src_col, tgt_col, prefix):
    def preprocess(examples):
        inputs  = [prefix + str(x) for x in examples[src_col]]
        targets = [str(x) for x in examples[tgt_col]]
        model_inputs = tokenizer(
            inputs, max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        label_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in lab]
            for lab in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs
    return preprocess

# =============================================================================
# 3. ENTRENAMIENTO
# =============================================================================

class EpochLogger(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = {k: v for k, v in (state.log_history[-1] if state.log_history else {}).items()}
        ep   = int(state.epoch)
        loss = logs.get("loss", "?")
        eloss = logs.get("eval_loss", "?")
        print(f"    Época {ep:>2} — train_loss: {loss}  eval_loss: {eloss}")

def train_kfold(df, tokenizer, src_col, tgt_col, prefix, task_name, output_base):
    banner(f"Entrenamiento: {task_name}")
    preprocess_fn = make_preprocess_fn(tokenizer, src_col, tgt_col, prefix)
    dataset = Dataset.from_pandas(df)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        train_sub = dataset.select(train_idx)
        val_sub   = dataset.select(val_idx)
        col_names = train_sub.column_names

        tok_train = train_sub.map(preprocess_fn, batched=True, remove_columns=col_names)
        tok_val   = val_sub.map(preprocess_fn,   batched=True, remove_columns=col_names)

        model     = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(device)
        ckpt_dir  = output_base / f"checkpoints_fold_{fold}"
        final_dir = output_base / f"final_model_fold_{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        args = Seq2SeqTrainingArguments(
    		output_dir                  = str(final_dir),
    		eval_strategy               = "epoch",
    		save_strategy               = "no",        # no guardar checkpoints
    		learning_rate               = LEARNING_RATE,
    		per_device_train_batch_size = BATCH_SIZE,
    		gradient_accumulation_steps = GRAD_ACCUM,
    		weight_decay                = 0.01,
    		warmup_steps                = 50,
    		num_train_epochs            = EPOCHS,
    		predict_with_generate       = True,
    		fp16                        = False,
    		logging_steps               = 20,
    		report_to                   = "none",
	)

	trainer = Seq2SeqTrainer(
	    	model            = model,
	    	args             = args,
    		train_dataset    = tok_train,
	    	eval_dataset     = tok_val,
    		processing_class = tokenizer,
	    	data_collator    = DataCollatorForSeq2Seq(
        		tokenizer, model=model, label_pad_token_id=-100
  		  ),
   		 callbacks        = [EpochLogger()],   # sin EarlyStoppingCallback
	)

        trainer.train()
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        best = trainer.state.best_metric
        fold_losses.append(best)
        print(f"Fold {fold + 1} guardado. Mejor eval_loss: {best:.4f}")

        del model
        torch.cuda.empty_cache()

    print(f"\nValidación cruzada — eval_loss por fold: {[f'{x:.4f}' for x in fold_losses]}")
    print(f"Media: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
# =============================================================================
# 4. INFERENCIA CON ENSEMBLE
# =============================================================================

def translate_ensemble(model_dir, texts, prefix, batch_size=8):
    fold_dirs = sorted([
        d for d in model_dir.iterdir()
        if d.is_dir() and d.name.startswith("final_model_fold_")
    ])
    if not fold_dirs:
        raise FileNotFoundError(f"No hay modelos en {model_dir}")

    print(f"Ensemble con {len(fold_dirs)} modelos desde '{model_dir.name}'")
    prefixed = [prefix + t for t in texts]
    all_preds = []

    for fold_dir in fold_dirs:
        print(f"  {fold_dir.name}...")
        tok = AutoTokenizer.from_pretrained(str(fold_dir), local_files_only=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            str(fold_dir), local_files_only=True
        ).to(device)
        mdl.eval()

        fold_preds = []
        for i in range(0, len(prefixed), batch_size):
            batch  = prefixed[i : i + batch_size]
            inputs = tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN
            ).to(device)
            with torch.no_grad():
                out = mdl.generate(
                    **inputs,
                    max_length        = MAX_LEN,
                    num_beams         = NUM_BEAMS,
                    early_stopping    = True,
                    no_repeat_ngram_size = 3,
                    length_penalty    = 1.0,
                )
            fold_preds.extend(tok.batch_decode(out, skip_special_tokens=True))

        all_preds.append(fold_preds)
        del mdl
        torch.cuda.empty_cache()

    # Ensemble: mayoría de votos; en empate, el más largo
    final = []
    for i in range(len(texts)):
        candidates = [all_preds[f][i] for f in range(len(fold_dirs))]
        best = max(set(candidates), key=candidates.count)
        if candidates.count(best) == 1:
            best = max(candidates, key=len)
        final.append(best)

    return final

# =============================================================================
# 5. EVALUACIÓN
# =============================================================================

def evaluate_predictions(preds, refs, label):
    try:
        import sacrebleu as sb
        bleu = sb.corpus_bleu(preds, [refs]).score
        chrf = sb.corpus_chrf(preds, [refs]).score
    except Exception as e:
        bleu, chrf = 0.0, 0.0
        print(f"  sacrebleu error: {e}")

    try:
        import evaluate as ev
        meteor_m = ev.load("meteor")
        meteor = meteor_m.compute(predictions=preds, references=refs)["meteor"] * 100
    except Exception as e:
        meteor = 0.0
        print(f"  meteor error: {e}")

    print(f"\n{label}")
    print(f"  BLEU:   {bleu:.2f}")
    print(f"  METEOR: {meteor:.2f}")
    print(f"  chrF:   {chrf:.2f}")
    return {"bleu": bleu, "meteor": meteor, "chrf": chrf}

# =============================================================================
# 6. SUBMISSION
# =============================================================================

def write_submission(preds, ids, path):
    with open(path, "w", encoding="utf-8") as f:
        for id_, pred in zip(ids, preds):
            f.write(f'"{id_}"\t"{pred}"\n')
    print(f"Submission: {path}  ({len(preds)} líneas)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline MSLG-SPA 2026")
    parser.add_argument(
        "--task",
        choices=["all", "train", "inference", "evaluate"],
        default="all",
        help="Qué parte del pipeline ejecutar",
    )
    args = parser.parse_args()

    check_hardware()
    make_dirs()

    # -- Datos --
    df_train        = load_train_data()
    df_test_m2s     = load_test_data("MSLG2SPA_test.txt", "MSLG")
    df_test_s2m     = load_test_data("SPA2MSLG_test.txt", "SPA")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # -- Entrenamiento --
    if args.task in ("all", "train"):
        train_kfold(
            df=df_train,
            tokenizer=tokenizer,
            src_col="MSLG", tgt_col="SPA",
            prefix=PREFIX_MSLG2SPA,
            task_name="mslg2spa",
            output_base=MODELS_DIR / "mslg2spa_final",
        )
        train_kfold(
            df=df_train,
            tokenizer=tokenizer,
            src_col="SPA", tgt_col="MSLG",
            prefix=PREFIX_SPA2MSLG,
            task_name="spa2mslg",
            output_base=MODELS_DIR / "spa2mslg_final",
        )

    # -- Inferencia --
    if args.task in ("all", "inference"):
        banner("Inferencia sobre test sets")

        m2s_preds = translate_ensemble(
            MODELS_DIR / "mslg2spa_final",
            df_test_m2s["MSLG"].tolist(),
            PREFIX_MSLG2SPA,
        )
        s2m_preds = translate_ensemble(
            MODELS_DIR / "spa2mslg_final",
            df_test_s2m["SPA"].tolist(),
            PREFIX_SPA2MSLG,
        )

        print("\nEjemplos MSLG → SPA:")
        for src, pred in zip(df_test_m2s["MSLG"].tolist()[:3], m2s_preds[:3]):
            print(f"  {src}  →  {pred}")

        print("\nEjemplos SPA → MSLG:")
        for src, pred in zip(df_test_s2m["SPA"].tolist()[:3], s2m_preds[:3]):
            print(f"  {src}  →  {pred}")

        write_submission(
            m2s_preds, df_test_m2s["ID"].tolist(),
            SUBMISSIONS / f"{TEAM}_{RUN}_MSLG2SPA.txt",
        )
        write_submission(
            s2m_preds, df_test_s2m["ID"].tolist(),
            SUBMISSIONS / f"{TEAM}_{RUN}_SPA2MSLG.txt",
        )

    # -- Evaluación interna --
    if args.task in ("all", "evaluate"):
        banner("Evaluación interna (último fold de validación)")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splits = list(kf.split(df_train))
        _, val_idx = splits[-1]
        df_val = df_train.iloc[val_idx]

        val_m2s = translate_ensemble(
            MODELS_DIR / "mslg2spa_final",
            df_val["MSLG"].tolist(), PREFIX_MSLG2SPA,
        )
        evaluate_predictions(val_m2s, df_val["SPA"].tolist(), "MSLG → SPA")

        val_s2m = translate_ensemble(
            MODELS_DIR / "spa2mslg_final",
            df_val["SPA"].tolist(), PREFIX_SPA2MSLG,
        )
        evaluate_predictions(val_s2m, df_val["MSLG"].tolist(), "SPA → MSLG")

    banner("Pipeline terminado")
    print(f"Submissions en: {SUBMISSIONS.resolve()}\n")


if __name__ == "__main__":
    main()

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_inference(model_dir, input_path, output_path):
    # Convertimos la ruta relativa a una ruta absoluta para evitar errores de Repo ID
    abs_model_dir = os.path.abspath(model_dir)
    print(f"Loading models from {abs_model_dir}...")
    
    # Usamos el path absoluto del Fold 0 para el tokenizer
    tokenizer_path = os.path.join(abs_model_dir, "final_model_fold_0")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    models = []
    for i in range(5):
        fold_path = os.path.join(abs_model_dir, f"final_model_fold_{i}")
        if os.path.exists(fold_path):
            print(f"Loading fold {i}...")
            models.append(AutoModelForSeq2SeqLM.from_pretrained(fold_path, local_files_only=True).to(device))
        else:
            print(f"Warning: {fold_path} not found. Skipping.")

    if not models:
        print("Error: No models were loaded.")
        return

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    predictions = []
    for text in lines:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # Generación usando el Fold 0
            output_tokens = models[0].generate(**inputs, max_length=128)
            decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        predictions.append(f"'{decoded}'")

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # MSLG -> SPA
    run_inference(
        "./models/mslg2spa_final", 
        "data/raw/MSLG_SPA_test_public.txt", 
        "mslg2spa_predicciones.txt"
    )
    
    # SPA -> MSLG
    run_inference(
        "./models/spa2mslg_final", 
        "data/raw/SPA_MSLG_test_public.txt", 
        "spa2mslg_predicciones.txt"
    )
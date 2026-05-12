import torch
from transformers import MarianTokenizer, MarianMTModel
import os
from pathlib import Path

# Usar la tarjeta de video si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_ensemble_inference(base_model_dir, input_path, output_path):
    # Buscar las carpetas de los 5 entrenamientos (folds)
    base_path = Path(base_model_dir).resolve()
    fold_dirs = [base_path / f"final_model_fold_{i}" for i in range(5)]
    
    # Revisar cuáles carpetas sí existen
    valid_folds = [d for d in fold_dirs if d.exists()]
    if not valid_folds:
        print("Error: No se encontraron los modelos.")
        return

    # Preparar el traductor de palabras (tokenizer)
    tokenizer = MarianTokenizer.from_pretrained(str(valid_folds[0]), local_files_only=True)

    if not os.path.exists(input_path):
        print(f"Error: No existe el archivo {input_path}")
        return

    # Leer las frases que vamos a traducir
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"Empezando: Se usaran {len(valid_folds)} modelos para {len(lines)} frases.")
    
    # Guardar las respuestas de cada modelo
    all_votes = [[] for _ in range(len(lines))]

    # Cargar y usar un modelo a la vez para no trabar la PC
    for fold_dir in valid_folds:
        print(f"Traduciendo con el modelo: {fold_dir.name}...")
        model = MarianMTModel.from_pretrained(str(fold_dir), local_files_only=True).to(device)
        model.eval()
        
        with torch.no_grad():
            for i, text in enumerate(lines):
                inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
                # Generar la traduccion
                output_tokens = model.generate(**inputs, max_length=128, num_beams=4)
                decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                # Guardar el "voto" de este modelo
                all_votes[i].append(decoded)
        
        # Quitar el modelo de la memoria antes de cargar el siguiente
        del model
        torch.cuda.empty_cache()

    # Elegir la respuesta en la que mas modelos coincidan
    print("Eligiendo la mejor respuesta por mayoria de votos...")
    final_predictions = []
    for votes in all_votes:
        # Quedarse con la frase que mas se repite
        most_common = max(set(votes), key=votes.count)
        final_predictions.append(f"'{most_common}'")

    # Guardar el archivo final en la carpeta raiz
    output_abs_path = Path(output_path).resolve()
    with open(output_abs_path, "w", encoding="utf-8") as f:
        for p in final_predictions:
            f.write(p + "\n")
    
    print(f"LISTO: Archivo guardado en {output_abs_path}")

if __name__ == "__main__":
    # Tarea: Glosa a Español
    run_ensemble_inference(
        "models/mslg2spa_final", 
        "data/raw/MSLG2SPA_test.txt", 
        "submissions/mslg2spa_predicciones.txt"
    )
    
    # Tarea: Español a Glosa
    run_ensemble_inference(
        "models/spa2mslg_final", 
        "data/raw/SPA2MSLG_test.txt", 
        "submissions/spa2mslg_predicciones.txt"
    )
    #Para validación
    run_ensemble_inference(
        "models/mslg2spa_final", 
        "data/raw/solo_mslg_entrada.txt", 
        "submissions/mslg2spa_train_eval_predicciones.txt"
    )
    #Para validación
    run_ensemble_inference(
        "models/spa2mslg_final", 
        "data/raw/solo_spa_entrada.txt", 
        "submissions/spa2mslg_train_eval_predicciones.txt"
    )
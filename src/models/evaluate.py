import sacrebleu

def calculate_metrics(preds_path, refs_path):
    with open(preds_path, "r", encoding="utf-8") as f:
        # Quitamos las comillas simples para evaluar el texto puro
        preds = [line.strip().replace("'", "") for line in f.readlines()]
    
    with open(refs_path, "r", encoding="utf-8") as f:
        refs = [line.strip() for line in f.readlines()]

    # BLEU SCORE
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print(f"--- Reporte de Metricas ---")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Detalles: {bleu.precisions}")

if __name__ == "__main__":
    calculate_metrics("submissions/mslg2spa_predicciones.txt", "data/processed/referencias_spa.txt")
    calculate_metrics("submissions/spa2mslg_predicciones.txt", "data/processed/referencias_mslg.txt")
    calculate_metrics("submissions/spa2mslg_train_eval_predicciones.txt", "data/processed/referencias_mslg.txt")
    calculate_metrics("submissions/mslg2spa_train_eval_predicciones.txt", "data/processed/referencias_spa.txt")
    
    
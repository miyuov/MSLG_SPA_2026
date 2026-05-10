# Este código crea archivos con UNA SOLA COLUMNA para que el modelo no se confunda
with open("data/raw/MSLG_SPA_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()[1:] 

with open("data/raw/solo_mslg_entrada.txt", "w", encoding="utf-8") as f_mslg, \
     open("data/raw/solo_spa_entrada.txt", "w", encoding="utf-8") as f_spa:
    
    for line in lines:
        partes = line.strip().split('\t')
        if len(partes) >= 3:
            # Guardamos la columna 1 (Glosas) para probar el traductor a Español
            f_mslg.write(partes[1] + "\n")
            # Guardamos la columna 2 (Español) para probar el traductor a Glosas
            f_spa.write(partes[2] + "\n")

print("Listo")
import os

def create_reference_files(input_file):
    # Leer el archivo original de entrenamiento
    with open(input_file, "r", encoding="utf-8") as f:
        # Saltamos la primera línea que es el encabezado (ID MSLG SPA)
        lines = f.readlines()[1:]

    # 1. Crear referencia para MSLG -> SPA (Necesitamos la columna SPA)
    with open("data/processed/referencias_spa.txt", "w", encoding="utf-8") as f_spa:
        for line in lines:
            parts = line.strip().split('\t')
            # La columna SPA es la tercera (índice 2)
            if len(parts) >= 3:
                f_spa.write(parts[2] + "\n")

    # 2. Crear referencia para SPA -> MSLG (Necesitamos la columna MSLG)
    with open("data/processed/referencias_mslg.txt", "w", encoding="utf-8") as f_mslg:
        for line in lines:
            parts = line.strip().split('\t')
            # La columna MSLG es la segunda (índice 1)
            if len(parts) >= 2:
                f_mslg.write(parts[1] + "\n")

    print("Se crearon 'referencias_spa.txt' y 'referencias_mslg.txt' en data/processed/")

# Ejecutar la función con tu archivo original
create_reference_files("data/raw/MSLG_SPA_train.txt")

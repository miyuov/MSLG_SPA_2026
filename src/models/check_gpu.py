import torch

def check_hardware():
    print("--- Verificación de Hardware para Ciencia de Datos ---")
    
    # Verifica si CUDA (el lenguaje de NVIDIA) esta disponible
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Memoria total en Gigabytes
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Detectada: {gpu_name}")
        print(f"Memoria VRAM Total: {total_mem:.2f} GB")
        
        # Criterio tecnico para el modelo
        if total_mem < 4:
            print("Estado: VRAM limitada. Usa batch_size=4 o 2.")
        elif total_mem < 8:
            print("Estado: VRAM aceptable. Puedes usar MarianMT sin problemas.")
        else:
            print("Estado: Excelente. Tienes potencia para modelos mas pesados.")
    else:
        print("GPU No detectada. El entrenamiento sera muy lento en CPU.")

if __name__ == "__main__":
    check_hardware()
# analyze_split_distribution.py

import json
import pandas as pd
import os
from collections import Counter

# --- CONFIGURACIÓN ---
# ¡Asegúrate de que estas rutas sean correctas!

# Ruta al archivo de mapeo que ya creaste
MAPPING_CSV_FILE = "layout_mapping.csv"

# Rutas a tus archivos de anotaciones
ANNOTATION_FILES = {
    "train": "/Users/jarylml/Desktop/PROYECTO/pklotdataset/train/_annotations.coco.json",
    "valid": "/Users/jarylml/Desktop/PROYECTO/pklotdataset/valid/_annotations.coco.json",
    "test": "/Users/jarylml/Desktop/PROYECTO/pklotdataset/test/_annotations.coco.json"
}

def analyze_distribution():
    """
    Analiza y muestra cuántas imágenes de cada layout hay en los sets
    de entrenamiento, validación y prueba.
    """
    if not os.path.exists(MAPPING_CSV_FILE):
        print(f"Error: No se encontró el archivo '{MAPPING_CSV_FILE}'.")
        print("Por favor, asegúrate de que el archivo de mapeo esté en la misma carpeta.")
        return

    # Cargar el mapeo de nombres de archivo a layouts
    mapping_df = pd.read_csv(MAPPING_CSV_FILE)
    filename_to_layout = pd.Series(mapping_df.layout_name.values, index=mapping_df.filename).to_dict()

    print("--- Análisis de Distribución del Dataset ---\n")
    
    results = {}
    total_counts = Counter()

    # Analizar cada split (train, valid, test)
    for split_name, json_path in ANNOTATION_FILES.items():
        if not os.path.exists(json_path):
            print(f"Advertencia: No se encontró el archivo de anotaciones para '{split_name}': {json_path}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Contar los layouts para este split
        layout_counts = Counter()
        for image in data['images']:
            filename = image['file_name']
            layout = filename_to_layout.get(filename, 'desconocido')
            layout_counts[layout] += 1
        
        results[split_name] = dict(layout_counts)
        total_counts.update(layout_counts)
    
    # Crear y mostrar una tabla de resultados
    summary_df = pd.DataFrame(results).fillna(0).astype(int)
    
    # Añadir una fila con el total
    summary_df.loc['TOTAL'] = summary_df.sum()
    
    print("Número de imágenes por layout y por conjunto de datos:")
    print(summary_df)
    
    # Dar un diagnóstico basado en la distribución
    print("\n--- Diagnóstico Preliminar ---")
    straight_wide_train_count = summary_df.loc['straight_wide', 'train']
    
    if straight_wide_train_count < 200:
        print(f"¡Atención! Solo tienes {straight_wide_train_count} imágenes de 'straight_wide' en tu conjunto de entrenamiento.")
        print("Este número es probablemente demasiado bajo y es la causa más probable del bajo rendimiento del modelo.")
        print("Recomendación: Mueve más imágenes 'straight_wide' a tu conjunto de 'train'.")
    else:
        print(f"Tienes {straight_wide_train_count} imágenes de 'straight_wide' en tu conjunto de entrenamiento.")
        print("Este número parece razonable. El problema podría estar en otro lugar, como la calidad de las anotaciones.")


if __name__ == "__main__":
    analyze_distribution()
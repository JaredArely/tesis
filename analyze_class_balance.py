# analyze_class_balance.py

import json
from collections import Counter

# --- CONFIGURACIÓN ---
# Ruta a tu archivo de anotaciones de entrenamiento
ANNOTATIONS_FILE_PATH = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/train/_annotations.coco.json"

def analyze_balance():
    """
    Analiza y muestra el balance de clases (ocupado vs. libre) en el
    conjunto de entrenamiento.
    """
    print("--- Analizando el Balance de Clases en el Set de Entrenamiento ---")
    try:
        with open(ANNOTATIONS_FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de anotaciones en '{ANNOTATIONS_FILE_PATH}'")
        return

    # Extraer los nombres de las categorías (ej. {1: 'space-occupied', 2: 'space-empty'})
    if 'categories' not in data:
        print("Error: No se encontró la sección 'categories' en el archivo JSON.")
        return
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Contar cuántas veces aparece cada category_id en las anotaciones
    if 'annotations' not in data:
        print("Error: No se encontró la sección 'annotations' en el archivo JSON.")
        return
    class_counts = Counter(ann['category_id'] for ann in data['annotations'])
    
    print("\nConteo total de anotaciones por categoría:")
    total = 0
    for cat_id, count in class_counts.items():
        cat_name = category_map.get(cat_id, f"ID Desconocido {cat_id}")
        print(f"  -> Categoría '{cat_name}': {count} anotaciones")
        total += count
    
    print(f"\nTotal de cajones anotados: {total}")
    
    # Diagnóstico
    if len(class_counts) == 2:
        counts = list(class_counts.values())
        ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        print(f"\nRatio entre la clase mayoritaria y la minoritaria: {ratio:.2f} a 1")
        if ratio > 3.0:
            print("¡Atención! Existe un desequilibrio de clases significativo (mayor a 3:1).")
            print("Esto podría ser una causa importante del mal rendimiento del modelo.")
        else:
            print("El balance de clases parece razonable (menor a 3:1).")

if __name__ == "__main__":
    analyze_balance()
    
import torch
import cv2
import os
import pickle
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn

# Importamos la configuración
try:
    from parking_config import ROI_COORDS, NUM_SPACES
except ImportError:
    print("Error: No se pudo importar la configuración desde 'parking_config.py'.")
    exit()

# Recreamos las funciones necesarias
def create_model_finetune(num_spaces, device):
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_spaces)
    return resnet18.to(device)

def get_conditions_from_filename(filename):
    """Recreamos esta función para poder agrupar por condición."""
    actual_filename = os.path.basename(filename)
    weather = 'sunny'
    if 'rainy' in actual_filename.lower():
        weather = 'rainy'
    time_of_day = 'day'
    match = re.search(r'_(\d{2})_\d{2}_\d{2}', actual_filename)
    if match and (int(match.group(1)) >= 18 or int(match.group(1)) < 6):
        time_of_day = 'night'
    return f"{weather}_{time_of_day}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    DATA_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/processed_data"
    
    transform_pred = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

    print("\nCargando modelo entrenado...")
    model = create_model_finetune(NUM_SPACES, device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    all_test_data = []
    for f in os.listdir(DATA_DIR):
        if f.startswith('test_') and f.endswith('.pkl'):
            with open(os.path.join(DATA_DIR, f), 'rb') as file:
                paths, labels = pickle.load(file)
                for i in range(len(paths)):
                    all_test_data.append({'path': paths[i], 'label': labels[i]})
    
    if not all_test_data:
        print("Error: No se encontraron datos de prueba para analizar.")
        return

    # Estructura para guardar las estadísticas
    stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total_images": 0, "images_with_errors": 0})

    print(f"\nAnalizando {len(all_test_data)} imágenes de prueba...")
    
    for data_point in all_test_data:
        image_path, ground_truth = data_point['path'], data_point['label']
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: continue
        
        condition = get_conditions_from_filename(image_path)
        stats[condition]["total_images"] += 1

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform_pred(image=image_rgb)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy() > 0.5

        has_error = False
        for i in range(len(predictions)):
            pred = predictions[i]
            truth = ground_truth[i] == 1.0

            if pred and truth: stats[condition]["tp"] += 1
            elif not pred and not truth: stats[condition]["tn"] += 1
            elif pred and not truth: stats[condition]["fp"] += 1; has_error = True
            elif not pred and truth: stats[condition]["fn"] += 1; has_error = True
        
        if has_error:
            stats[condition]["images_with_errors"] += 1

    print("\n--- Análisis de Errores Completado ---")
    
    results = []
    for condition, data in stats.items():
        tp, tn, fp, fn = data["tp"], data["tn"], data["fp"], data["fn"]
        total_spaces = tp + tn + fp + fn
        if total_spaces == 0: continue

        accuracy = (tp + tn) / total_spaces * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        results.append({
            "Condición": condition.replace("_", " ").title(),
            "Precisión (%)": precision,
            "Recall (Sensibilidad) (%)": recall,
            "Especificidad (%)": specificity,
            "Exactitud General (%)": accuracy,
            "Falsos Positivos (FP)": fp,
            "Falsos Negativos (FN)": fn,
            "Imágenes con Error": f"{data['images_with_errors']}/{data['total_images']}"
        })

    if results:
        summary_df = pd.DataFrame(results).set_index('Condición')
        print(summary_df.to_string(formatters={
            'Precisión (%)':'{:.2f}'.format,
            'Recall (Sensibilidad) (%)':'{:.2f}'.format,
            'Especificidad (%)':'{:.2f}'.format,
            'Exactitud General (%)':'{:.2f}'.format
        }))
    else:
        print("No se generaron estadísticas.")

if __name__ == '__main__':
    main()
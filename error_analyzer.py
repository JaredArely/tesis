import torch
import cv2
import os
import pickle
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn

try:
    from parking_config import ROI_COORDS, NUM_SPACES
except ImportError:
    print("Error: No se pudo importar la configuración desde 'parking_config.py'.")
    exit()


def create_model_finetune(num_spaces, device):
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_spaces)
    return resnet18.to(device)

def visualize_errors(image_path, predictions, labels, rois):
    """
    Visualiza los falsos positivos (FP) y falsos negativos (FN)
    en una imagen, usando diferentes colores y devuelve el conteo de errores.

    FP (Falso Positivo): El modelo predijo OCUPADO, pero el espacio está LIBRE. -> ROJO
    FN (Falso Negativo): El modelo predijo LIBRE, pero el espacio está OCUPADO. -> VERDE
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return None, 0

    output_image = image.copy()
    error_count = 0  # <--- INICIALIZACIÓN DEL CONTADOR
    
    # Colores BGR para OpenCV
    COLOR_FP = (0, 0, 255)  # ROJO para Falsos Positivos
    COLOR_FN = (0, 255, 0)  # VERDE para Falsos Negativos
    COLOR_TP_TN = (200, 200, 200) # Gris para aciertos 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    # Asegurarse de que predictions y labels son arrays de numpy y tienen la misma longitud que rois
    # Las predicciones deben ser binarizadas (0 o 1) antes de esta función si no lo fueron antes
    predictions = (np.array(predictions).flatten() > 0.5).astype(int)
    labels = np.array(labels).flatten().astype(int)

    for i, roi_coords in enumerate(rois):
        if i >= len(predictions) or i >= len(labels):
            continue 

        # Aseguramos que las coordenadas sean enteros
        x, y, w, h = map(int, roi_coords)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        pred = predictions[i]
        label = labels[i]

        # Falso Positivo: Predijo OCUPADO (1), pero es LIBRE (0)
        if pred == 1 and label == 0:
            cv2.rectangle(output_image, (x1, y1), (x2, y2), COLOR_FP, thickness + 1)
            cv2.putText(output_image, "FP", (x1, y1 - 10), font, font_scale, COLOR_FP, thickness)
            error_count += 1
        
        # Falso Negativo: Predijo LIBRE (0), pero es OCUPADO (1)
        elif pred == 0 and label == 1:
            cv2.rectangle(output_image, (x1, y1), (x2, y2), COLOR_FN, thickness + 1)
            cv2.putText(output_image, "FN", (x1, y1 - 10), font, font_scale, COLOR_FN, thickness)
            error_count += 1
        
        # Opcional: Para visualizar aciertos (Verdaderos Positivos/Negativos)
        elif pred == label:
            cv2.rectangle(output_image, (x1, y1), (x2, y2), COLOR_TP_TN, 1) # Delgado y gris
            cv2.putText(output_image, "OK", (x1, y1 - 5), font, font_scale * 0.7, COLOR_TP_TN, 1)

    return output_image, error_count # <--- DEVOLUCIÓN DE IMAGEN Y CONTEO

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    DATA_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/processed_data"
    OUTPUT_ERROR_DIR = "error_analysis"
    if not os.path.exists(OUTPUT_ERROR_DIR):
        os.makedirs(OUTPUT_ERROR_DIR)

    transform_pred = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

    print("\nCargando modelo entrenado...")
    model = create_model_finetune(NUM_SPACES, device)
    
    # Manejo de error si 'best_model.pth' no existe (importante para el reinicio)
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo entrenado '{model_path}'. Asegúrate de ejecutar entrenamientonuevo.py primero.")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_test_data = []
    for f in os.listdir(DATA_DIR):
        # Aseguramos que solo tomamos los archivos de PRUEBA
        if f.startswith('test_') and f.endswith('.pkl'):
            with open(os.path.join(DATA_DIR, f), 'rb') as file:
                paths, labels = pickle.load(file)
                for i in range(len(paths)):
                    all_test_data.append({'path': paths[i], 'label': labels[i]})
    
    if not all_test_data:
        print("Advertencia: No se encontraron datos de prueba. Asegúrate de ejecutar editcat.py.")
        exit()

    print(f"\nAnalizando {len(all_test_data)} imágenes de prueba...")
    total_images_with_errors = 0
    
    for data_point in all_test_data:
        image_path, ground_truth = data_point['path'], data_point['label']
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: continue
        
        # Necesitamos la imagen RGB para el tensor y BGR para la visualización
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
        image_tensor = transform_pred(image=image_rgb)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            # Obtenemos las probabilidades (logits) para binarizar después
            predictions_logits = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Corregido el flujo: Ahora visualize_errors devuelve el conteo
        output_image, error_count = visualize_errors(image_path, predictions_logits, ground_truth, ROI_COORDS)
        
        if error_count > 0:
            total_images_with_errors += 1
            # Guardamos la imagen en la carpeta de análisis de errores
            output_filename = os.path.join(OUTPUT_ERROR_DIR, os.path.basename(image_path))
            cv2.imwrite(output_filename, output_image)
            print(f" -> ¡Error encontrado en '{os.path.basename(image_path)}'! ({error_count} fallos). Imagen guardada.")

    print(f"\nAnálisis completado. Se encontraron errores en {total_images_with_errors} de {len(all_test_data)} imágenes. (Revisa la carpeta '{OUTPUT_ERROR_DIR}')")

if __name__ == '__main__':
    main()
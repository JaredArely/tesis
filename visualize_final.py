import torch
import cv2
import os
import pickle
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn

try:
    from parking_config import ROI_COORDS, NUM_SPACES
except ImportError:
    print("Error: No se pudo importar 'ROI_COORDS' y 'NUM_SPACES' desde 'parking_config.py'.")
    exit()

def create_model_finetune(num_spaces, device):
    """Crea la arquitectura del modelo para poder cargar los pesos."""
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_spaces)
    return resnet18.to(device)

def visualize_predictions(image_path, predictions, layout_coords):
    """
    Dibuja únicamente los recuadros de color sobre una imagen.
    - ROJO: Predicción de Ocupado.
    - VERDE: Predicción de Libre.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error al cargar imagen: {image_path}")
        return None
        
    for i, roi in enumerate(layout_coords):
        x, y, w, h = map(int, roi)
        pred_is_occupied = predictions[i] > 0.5
        
        # Asignar color basado en la predicción
        if pred_is_occupied:
            color = (0, 0, 255)  # Rojo para Ocupado
        else:
            color = (0, 255, 0)  # Verde para Libre
            
        # Dibujar SOLO el rectángulo, sin texto
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        
    return image_bgr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    DATA_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/processed_data"
    
    transform_pred = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    print("\nCargando modelo entrenado...")
    model = create_model_finetune(NUM_SPACES, device)
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo '{model_path}'.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_test_data = []
    for f in os.listdir(DATA_DIR):
        if f.startswith('test_') and f.endswith('.pkl'):
            with open(os.path.join(DATA_DIR, f), 'rb') as file:
                paths, _ = pickle.load(file)
                for path in paths:
                    all_test_data.append({'path': path})
    
    if not all_test_data:
        print("Error: No se encontraron datos de prueba.")
        return
    
    images_to_show = random.sample(all_test_data, min(10, len(all_test_data)))
    
    for i, data_point in enumerate(images_to_show):
        image_path = data_point['path']
        print(f"\nVisualizando imagen {i+1}/{len(images_to_show)}: {os.path.basename(image_path)}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform_pred(image=image_rgb)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()

        output_image = visualize_predictions(image_path, predictions, ROI_COORDS)
        
        if output_image is not None:
            cv2.imshow(f"Visualizacion de Estado ({i+1}/{len(images_to_show)})", output_image)
            print("  -> Presiona cualquier tecla en la ventana para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("\nVisualización completada.")

if __name__ == '__main__':
    main()
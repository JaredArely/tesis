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

def visualize_errors(image_path, predictions, ground_truth, layout_coords):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None: return None, 0
    
    error_count = 0
    for i, roi in enumerate(layout_coords):
        pred_is_occupied = predictions[i] > 0.5
        truth_is_occupied = ground_truth[i] > 0.5
        
        if pred_is_occupied != truth_is_occupied:
            error_count += 1
            color = (0, 0, 255) # Rojo para Error
            pred_label = "Ocupado" if pred_is_occupied else "Libre"
            truth_label = "Ocupado" if truth_is_occupied else "Libre"
            label = f"P:{pred_label} / V:{truth_label}"
            x, y, w, h = map(int, roi)
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_bgr, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return image_bgr, error_count

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
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    all_test_data = []
    for f in os.listdir(DATA_DIR):
        if f.startswith('test_') and f.endswith('.pkl'):
            with open(os.path.join(DATA_DIR, f), 'rb') as file:
                paths, labels = pickle.load(file)
                for i in range(len(paths)):
                    all_test_data.append({'path': paths[i], 'label': labels[i]})
    
    print(f"\nAnalizando {len(all_test_data)} imágenes de prueba...")
    total_images_with_errors = 0
    
    for data_point in all_test_data:
        image_path, ground_truth = data_point['path'], data_point['label']
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = transform_pred(image=image_rgb)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()

        output_image, error_count = visualize_errors(image_path, predictions, ground_truth, ROI_COORDS)
        
        if error_count > 0:
            total_images_with_errors += 1
            output_filename = os.path.join(OUTPUT_ERROR_DIR, os.path.basename(image_path))
            cv2.imwrite(output_filename, output_image)
            print(f" -> ¡Error encontrado en '{os.path.basename(image_path)}'! ({error_count} fallos). Imagen guardada.")

    print(f"\nAnálisis completado. Se encontraron errores en {total_images_with_errors} de {len(all_test_data)} imágenes.")

if __name__ == '__main__':
    main()
import torch
import cv2
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

# --- CONFIGURACIÓN DE VIDEO ---
# Para usar un archivo de video, ruta: "ruta/video.mp4"
# Para usar la cámara web, índice: 0
VIDEO_SOURCE = 0

def create_model_finetune(num_spaces, device):
    """Crea la arquitectura del modelo para poder cargar los pesos."""
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_spaces)
    return resnet18.to(device)

def draw_predictions_on_frame(frame, predictions, layout_coords):
    """Dibuja los recuadros de color (Rojo=Ocupado, Verde=Libre) sobre un fotograma."""
    for i, roi in enumerate(layout_coords):
        pred_is_occupied = predictions[i] > 0.5
        color = (0, 0, 255) if pred_is_occupied else (0, 255, 0)
        x, y, w, h = map(int, roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    transform_pred = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

    print("\nCargando modelo entrenado...")
    model = create_model_finetune(NUM_SPACES, device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # Abrir la fuente de video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video: {VIDEO_SOURCE}")
        return

    print("\nIniciando detección en video... Presiona 'q' en la ventana para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el fotograma.")
            break

        # Preparar el fotograma para la predicción
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform_pred(image=image_rgb)['image'].unsqueeze(0).to(device)

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Dibujar las predicciones sobre el fotograma
        frame_with_predictions = draw_predictions_on_frame(frame, predictions, ROI_COORDS)
        
        # Mostrar el resultado
        cv2.imshow("Detector de Estacionamiento en Tiempo Real", frame_with_predictions)

        # Condición de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    main()
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam, AdamW
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from parking_config import NUM_SPACES

class ParkingDatasetFromPaths(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: image_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else: image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image_rgb)['image']
        return image, torch.tensor(label, dtype=torch.float32)

#transformaciones 
transform_train = A.Compose([A.Resize(224, 224), A.Rotate(limit=5, p=0.5), A.RandomBrightnessContrast(p=0.75), A.GaussNoise(p=0.5), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
transform_val_test = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

# creación de modelo, entrenamiento y evaluación 
def create_model_finetune(num_spaces, device):
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_spaces)
    return resnet18.to(device)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train(); total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(images); loss = loss_fn(outputs, labels)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval(); total_loss = 0; all_preds_flat, all_labels_flat = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); loss = loss_fn(outputs, labels); total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds_flat.append(preds.cpu().numpy().flatten())
            all_labels_flat.append(labels.cpu().numpy().flatten())
    if not all_labels_flat or len(all_labels_flat[0]) == 0: return 0, np.array([]), np.array([])
    return total_loss/len(dataloader), np.concatenate(all_labels_flat), np.concatenate(all_preds_flat)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Usando dispositivo: {device}")
    DATA_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/processed_data"
    
    conditions = ["sunny_day", "rainy_day", "sunny_night", "rainy_night"]
    
    all_train_paths, all_train_labels_list = [], []
    for c in conditions:
        f_path = os.path.join(DATA_DIR, f"train_{c}.pkl")
        if os.path.exists(f_path):
            with open(f_path, 'rb') as f: paths, labels = pickle.load(f); all_train_paths.extend(paths); all_train_labels_list.append(labels)

    all_valid_paths, all_valid_labels_list = [], []
    for c in conditions:
        f_path = os.path.join(DATA_DIR, f"valid_{c}.pkl")
        if os.path.exists(f_path):
            with open(f_path, 'rb') as f: paths, labels = pickle.load(f); all_valid_paths.extend(paths); all_valid_labels_list.append(labels)

    if not all_train_paths: print("No se encontraron datos de entrenamiento. Ejecuta editcat.py primero."); return
        
    train_labels_np = np.concatenate(all_train_labels_list, axis=0) if all_train_labels_list else np.array([])
    valid_labels_np = np.concatenate(all_valid_labels_list, axis=0) if all_valid_labels_list else np.array([])
        
    train_dataset = ParkingDatasetFromPaths(all_train_paths, train_labels_np, transform=transform_train)
    valid_dataset = ParkingDatasetFromPaths(all_valid_paths, valid_labels_np, transform=transform_val_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    model = create_model_finetune(NUM_SPACES, device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
    best_val_loss, epochs = float('inf'), 30

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"Época {epoch+1}/{epochs} | Pérdida Train: {train_loss:.4f} | Pérdida Val: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; torch.save(model.state_dict(), 'best_model.pth'); print("  -> ¡Mejor modelo guardado!")

    print("\n--- Evaluando el mejor modelo ---")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    results = []
    for c in conditions:
        f_path = os.path.join(DATA_DIR, f"test_{c}.pkl")
        if not os.path.exists(f_path): continue
        with open(f_path, 'rb') as f: test_paths, test_labels_np = pickle.load(f)
        if len(test_paths) == 0: continue
        
        test_dataset = ParkingDatasetFromPaths(test_paths, test_labels_np, transform=transform_val_test)
        test_loader = DataLoader(test_dataset, batch_size=16)
        test_loss, test_labels_flat, test_preds_flat = evaluate(model, test_loader, loss_fn, device)
        if test_labels_flat.size == 0: continue
        results.append({"Condición": c.replace("_", " ").title(), "Precisión (%)": precision_score(test_labels_flat, test_preds_flat, zero_division=0) * 100, "Recall (%)": recall_score(test_labels_flat, test_preds_flat, zero_division=0) * 100, "F1-Score (%)": f1_score(test_labels_flat, test_preds_flat, zero_division=0) * 100, "Exactitud (%)": accuracy_score(test_labels_flat, test_preds_flat) * 100})
    
    if results:
        df_results = pd.DataFrame(results).set_index('Condición')
        print("\n--- Resultados Finales ---")
        
        print(df_results.to_string(formatters={
            'Precisión (%)': '{:.2f}'.format,
            'Recall (%)': '{:.2f}'.format,
            'F1-Score (%)': '{:.2f}'.format,
            'Exactitud (%)': '{:.2f}'.format
        }))

        ax = df_results[['Precisión (%)', 'Recall (%)', 'F1-Score (%)']].plot(kind='bar', figsize=(12, 7), rot=0)
        ax.set_title('Rendimiento del Modelo por Condición')
        ax.set_ylabel('Puntuación (%)')
        ax.legend(title='Métrica')
        plt.tight_layout()
        plt.savefig('results_metrics.png')
        plt.close()
        print("\nGráfica de resultados guardada como 'results_metrics.png'")

if __name__ == "__main__":
    main()
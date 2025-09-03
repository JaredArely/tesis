import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns # Librería para gráficos estadísticos atractivos
import warnings

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
DATA_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/processed_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
K_FOLDS = 10
NUM_EPOCHS_PER_FOLD = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# --- CLASES Y FUNCIONES AUXILIARES ---
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

transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

def create_model(model_name, num_spaces, device):
    if model_name == 'ResNet-18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_spaces)
    elif model_name == 'ResNet-34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_spaces)
    elif model_name == 'MobileNetV3':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_spaces)
    else:
        raise ValueError("Nombre de modelo no reconocido")
    return model.to(device)

def train_and_evaluate_fold(model, train_loader, val_loader, device):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(NUM_EPOCHS_PER_FOLD):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(images); loss = loss_fn(outputs, labels)
            loss.backward(); optimizer.step()
    model.eval()
    all_preds_flat, all_labels_flat = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds_flat.append(preds.cpu().numpy().flatten())
            all_labels_flat.append(labels.cpu().numpy().flatten())
    if not all_labels_flat: return 0.0
    return f1_score(np.concatenate(all_labels_flat), np.concatenate(all_preds_flat), zero_division=0)

def main():
    print(f"Usando dispositivo: {DEVICE}")
    print("Iniciando proceso de Validación Cruzada (k=10). Esto puede tardar...")
    all_paths, all_labels_list = [], []
    for split in ['train', 'valid', 'test']:
        for condition in ["sunny_day", "rainy_day", "sunny_night", "rainy_night"]:
            f_path = os.path.join(DATA_DIR, f"train_{condition}.pkl") # Corregido para tomar de 'train' y no de 'split'
            if os.path.exists(f_path) and f_path not in all_paths: # Evitar duplicados
                 with open(f_path, 'rb') as f:
                    paths, labels = pickle.load(f)
                    all_paths.extend(paths)
                    all_labels_list.append(labels)
    if not all_paths: print("Error: No se encontraron datos."); return
    all_labels = np.concatenate(all_labels_list, axis=0)
    num_spaces = all_labels.shape[1]
    
    models_to_compare = ['ResNet-18', 'ResNet-34', 'MobileNetV3']
    results = {name: [] for name in models_to_compare}
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_paths)):
        print(f"\n--- Procesando Pliegue (Fold) {fold + 1}/{K_FOLDS} ---")
        train_paths, val_paths = np.array(all_paths)[train_idx], np.array(all_paths)[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]
        train_dataset = ParkingDatasetFromPaths(train_paths, train_labels, transform=transform)
        val_dataset = ParkingDatasetFromPaths(val_paths, val_labels, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        for model_name in models_to_compare:
            print(f"  Entrenando modelo: {model_name}...")
            model = create_model(model_name, num_spaces, DEVICE)
            f1 = train_and_evaluate_fold(model, train_loader, val_loader, DEVICE)
            results[model_name].append(f1)
            print(f"    -> F1-Score para {model_name} en este pliegue: {f1:.4f}")

    print("\n\n" + "="*60 + "\nANÁLISIS ESTADÍSTICO FINAL\n" + "="*60)
    results_df = pd.DataFrame(results)
    print("\nResultados de F1-Score en los 10 Pliegues:\n", results_df.to_string())
    
    print("\n--- 1. Análisis Descriptivo y de Normalidad ---")
    summary_stats = []
    for model_name in models_to_compare:
        scores = results_df[model_name]
        shapiro_test = stats.shapiro(scores)
        summary_stats.append({"Modelo": model_name, "Media F1": scores.mean(), "Std F1": scores.std(), "Skewness": stats.skew(scores), "Kurtosis": stats.kurtosis(scores), "Shapiro-Wilk (p-valor)": shapiro_test.pvalue})
    summary_df = pd.DataFrame(summary_stats).set_index("Modelo")
    print(summary_df.to_string())
    print("\nInterpretación Shapiro-Wilk: Si p-valor > 0.05, los datos pueden considerarse normales.")

    print("\n--- 2. Prueba de Homogeneidad de Varianzas (Levene) ---")
    levene_test = stats.levene(*[results_df[name] for name in models_to_compare])
    print(f"Estadístico de Levene: {levene_test.statistic:.4f}, p-valor: {levene_test.pvalue:.4f}")
    if levene_test.pvalue > 0.05: print("Interpretación: Las varianzas son homogéneas (iguales). Se puede usar ANOVA.")
    else: print("Interpretación: Las varianzas no son homogéneas. Se recomienda usar una prueba no paramétrica como Friedman.")

    print("\n--- 3. Prueba de Comparación de Modelos ---")
    friedman_test = stats.friedmanchisquare(*[results_df[name] for name in models_to_compare])
    print("\nPrueba de Friedman (compara los 3 modelos):")
    print(f"Estadístico de Friedman: {friedman_test.statistic:.4f}, p-valor: {friedman_test.pvalue:.4f}")
    if friedman_test.pvalue < 0.05: print("Conclusión: Existe una diferencia estadísticamente significativa en el rendimiento de al menos uno de los modelos.")
    else: print("Conclusión: No hay evidencia estadística para decir que un modelo es superior a otro.")

    # --- ### NUEVO: GENERACIÓN DE GRÁFICOS ### ---
    print("\n--- 4. Generando Gráficos de Resultados ---")
    
    # Preparamos los datos para Seaborn: de formato ancho a largo
    data_long = results_df.melt(var_name='Modelo', value_name='F1-Score')

    # Gráfico 1: Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Modelo', y='F1-Score', data=data_long)
    plt.title('Comparación de Modelos con Box Plot (10 Pliegues)')
    plt.ylabel('Puntuación F1')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('boxplot_resultados.png')
    plt.close()
    print(" -> Gráfico 'boxplot_resultados.png' guardado.")

    # Gráfico 2: Bar Chart con Barras de Error
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Modelo', y='F1-Score', data=data_long, capsize=0.1)
    plt.title('Rendimiento Promedio de Modelos con Desviación Estándar')
    plt.ylabel('Puntuación F1 Promedio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('barchart_resultados.png')
    plt.close()
    print(" -> Gráfico 'barchart_resultados.png' guardado.")
    # --- ### FIN DE LA SECCIÓN DE GRÁFICOS ### ---

if __name__ == "__main__":
    main()
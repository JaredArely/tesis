import pickle
import os

def load_main_layout(layout_file="layouts/layout_straight_wide.pkl"):
    """Carga el único layout que usaremos."""
    if not os.path.exists(layout_file):
        print(f"ADVERTENCIA: El archivo de layout '{layout_file}' no existe.")
        print("Asegúrate de haber ejecutado primero 'extract_master_layouts.py'.")
        return []
    
    with open(layout_file, 'rb') as f:
        coords = pickle.load(f)
    
    print(f"Layout 'straight_wide' cargado con {len(coords)} espacios.")
    return coords

# Coordenadas y el número de espacios como variables globales.
# Estas son las variables que 'editcat.py' necesita importar.
ROI_COORDS = load_main_layout()
NUM_SPACES = len(ROI_COORDS)
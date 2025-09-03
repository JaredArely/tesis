# audit_master_layout.py

import cv2
import pickle
import os

# --- CONFIGURACIÓN ---
# Ruta a la carpeta que contiene las imágenes de prueba
IMAGE_DIR = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/test/"

# Ruta al archivo de layout que queremos auditar
LAYOUT_FILE = "layouts/layout_straight_wide.pkl"

# --- IMÁGENES PARA AUDITAR ---
# Añade aquí los nombres de varios archivos de imagen 'straight_wide' de tu carpeta 'test'
# para ver si el mapa se alinea bien en todas ellas.
FILENAMES_TO_CHECK = [
    "2012-11-20_12_09_40_jpg.rf.e72f5d398687ab0806c1963bdfaaf1d8.jpg",
    "2012-11-20_13_24_41_jpg.rf.218158ae10ca32d9739cfd5b4fdf82fc.jpg",
    "2012-11-11_18_34_18_jpg.rf.6ba644d82e3f0015bc08162c0e2f89f0.jpg"
]

def audit_layout():
    """Dibuja los ROIs del mapa maestro sobre varias imágenes para verificar la alineación."""
    if not os.path.exists(LAYOUT_FILE):
        print(f"Error: No se encontró el archivo de layout '{LAYOUT_FILE}'.")
        print("Ejecuta 'extract_master_layouts.py' primero.")
        return

    # Cargar los ROIs del mapa maestro
    with open(LAYOUT_FILE, 'rb') as f:
        master_rois = pickle.load(f)
    
    print(f"Cargado mapa maestro con {len(master_rois)} ROIs.")
    
    for filename in FILENAMES_TO_CHECK:
        image_path = os.path.join(IMAGE_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Advertencia: No se pudo cargar la imagen '{filename}'")
            continue

        # Dibujar cada ROI del mapa maestro en la imagen
        for roi in master_rois:
            x, y, w, h = map(int, roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Dibujar en azul

        print(f"Mostrando auditoría para: {filename}. Presiona cualquier tecla para continuar...")
        
        scale = 800 / image.shape[1] 
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        cv2.imshow("Auditoria de Mapa Maestro", image)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\nAuditoría completada.")

if __name__ == "__main__":
    audit_layout()
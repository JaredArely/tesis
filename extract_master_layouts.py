import json
import pickle
import os

# --- CONFIGURACIÓN ---
ANNOTATIONS_FILE_PATH = "/Users/jarylml/Desktop/PROYECTO/pklotdataset/train/_annotations.coco.json"
OUTPUT_LAYOUTS_DIR = "layouts"

# imagen de referencia
REFERENCE_IMAGES = {
    "straight_wide": "2012-09-18_14_10_09_jpg.rf.bfaab2c2ebca5d4af81a45f44e391803.jpg"
}

def extract_and_save_layouts():
    print("Cargando archivo de anotaciones...")
    try:
        with open(ANNOTATIONS_FILE_PATH, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de anotaciones en '{ANNOTATIONS_FILE_PATH}'")
        return

    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    if not os.path.exists(OUTPUT_LAYOUTS_DIR):
        os.makedirs(OUTPUT_LAYOUTS_DIR)

    print("\nExtrayendo layout desde la imagen de referencia...")
    for layout_name, ref_filename in REFERENCE_IMAGES.items():
        print(f"--- Procesando layout: {layout_name} ---")
        image_id = filename_to_id.get(ref_filename)
        if image_id is None:
            print(f"  -> Error: No se encontró la imagen de referencia '{ref_filename}' en el JSON.")
            continue
        
        boxes = [ann['bbox'] for ann in annotations_by_image_id.get(image_id, [])]
        if not boxes:
            print(f"  -> Error: No se encontraron anotaciones para '{ref_filename}'.")
            continue
            
        sorted_boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
        output_path = os.path.join(OUTPUT_LAYOUTS_DIR, f"layout_{layout_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(sorted_boxes, f)
        print(f"  -> ¡Éxito! Layout '{layout_name}' guardado en '{output_path}'")

if __name__ == '__main__':
    extract_and_save_layouts()
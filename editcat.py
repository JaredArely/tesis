import json
import os
import numpy as np
import pickle
import re
from collections import defaultdict
from parking_config import ROI_COORDS, NUM_SPACES

def get_layout_name_from_filename(filename):
    """
    Analiza el nombre del archivo para determinar el tipo de disposición del estacionamiento.
    Devuelve 'zigzag', 'straight_wide', o 'ignorar'.
    """
    # diciembre de 2012 es 'zigzag'.
    if "2012-12" in filename:
        return "zigzag"
    elif "2012" in filename:
        return "straight_wide"
    return "ignorar"

def get_conditions_from_filename(filename):
    """Analiza el nombre del archivo para determinar el clima y la hora del día."""
    actual_filename = filename[0] if isinstance(filename, list) else filename
    weather = 'sunny'
    if 'rainy' in actual_filename.lower():
        weather = 'rainy'
    time_of_day = 'day'
    match = re.search(r'_(\d{2})_\d{2}_\d{2}', actual_filename)
    if match and (int(match.group(1)) >= 18 or int(match.group(1)) < 6):
        time_of_day = 'night'
    return f"{weather}_{time_of_day}"

def calculate_iou(boxA, boxB):
    """Calcula la Intersección sobre Unión (IoU) de dos bounding boxes."""
    boxA_coords = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB_coords = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
    xA = max(boxA_coords[0], boxB_coords[0])
    yA = max(boxA_coords[1], boxB_coords[1])
    xB = min(boxA_coords[2], boxB_coords[2])
    yB = min(boxA_coords[3], boxB_coords[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA_coords[2] - boxA_coords[0]) * (boxA_coords[3] - boxA_coords[1])
    boxBArea = (boxB_coords[2] - boxB_coords[0]) * (boxB_coords[3] - boxB_coords[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return 0.0 if denominator == 0 else interArea / denominator

def main_processing(base_folder_path, output_folder, data_split_name):
    """Procesa un conjunto de datos (train, valid, test)."""
    annotation_file_path = os.path.join(base_folder_path, "_annotations.coco.json")
    if not os.path.exists(annotation_file_path):
        return

    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    OCCUPIED_ID = 2
    
    image_id_to_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
        
    data_buckets = defaultdict(lambda: {'paths': [], 'labels': []})
    
    print(f"\n--- Procesando imágenes en: {base_folder_path} ---")
    for image_info in coco_data['images']:
        image_filename = image_info['file_name']
        
        if get_layout_name_from_filename(image_filename) != "straight_wide":
            continue

        image_path = os.path.join(base_folder_path, image_filename)
        if not os.path.exists(image_path):
            continue

        condition_name = get_conditions_from_filename(image_filename)
        occupancy_labels = np.zeros(NUM_SPACES, dtype=np.float32)
        annotations_for_image = image_id_to_annotations.get(image_info['id'], [])

        for i, roi_coords in enumerate(ROI_COORDS):
            best_iou, best_cat_id = 0, -1
            for ann in annotations_for_image:
                iou = calculate_iou(roi_coords, ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_cat_id = ann['category_id']
            
            if best_iou > 0.5 and best_cat_id == OCCUPIED_ID:
                occupancy_labels[i] = 1.0
        
        data_buckets[condition_name]['paths'].append(image_path)
        data_buckets[condition_name]['labels'].append(occupancy_labels)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for condition, data in data_buckets.items():
        if data['paths']:
            output_filename = f"{data_split_name}_{condition}.pkl"
            output_filepath = os.path.join(output_folder, output_filename)
            # Esta línea es la importante: guarda una tupla (dos objetos)
            with open(output_filepath, "wb") as f:
                pickle.dump((data['paths'], np.array(data['labels'], dtype=np.float32)), f)
            print(f"-> Guardado: {output_filename} ({len(data['paths'])} imágenes)")

if __name__ == "__main__":
    DATASET_BASE_PATH = "/Users/jarylml/Desktop/PROYECTO/pklotdataset"
    OUTPUT_DATA_DIR = os.path.join(DATASET_BASE_PATH, "processed_data")
    for split in ["train", "valid", "test"]:
        main_processing(os.path.join(DATASET_BASE_PATH, split), OUTPUT_DATA_DIR, split)
    print("\n--- ¡Proceso de preprocesamiento de datos finalizado! ---")
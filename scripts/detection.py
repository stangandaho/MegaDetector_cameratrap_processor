from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils
from pathlib import Path
import json
import torch
from PIL import Image
import numpy as np

# Set device 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set model weight (for MegaDetector6 - YOLOv9)
# DOWNLOAD_WEIGHT = 'https://zenodo.org/records/11192829/files/MDV6b-yolov9c.pt?download=1'
# MODEL_PATH = r'D:\Python Projects\megadetector\v6\MDV6b-yolov9c.pt'

detection_model = pw_detection.MegaDetectorV5(device=DEVICE)

# Image path
img_path = r"./demo_data/10200020.JPG"

# Function to get result
def get_results(image_path, conf_thres = 0.55):
  # Opening and converting the image to RGB format
  img = np.array(Image.open(image_path).convert("RGB"))

  # Initializing the Yolo-specific transform for the image
  transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                stride=detection_model.STRIDE)

  # Performing the detection on the single image
  results = detection_model.single_image_detection(transform(img), img.shape, image_path, conf_thres=conf_thres)
  return results


# Count specific class
def class_number(results = dict, category = str):
  l = []
  detect_category = [x.split() for x in results.get('labels')]
  detect_category = [x[0] for x in detect_category]
  for x in detect_category:
    if x == category:
      l.append(x)

  return len(l)


# Function to get unique category from detection class
def get_unique(list):
  l = []
  for x in list:
    if x not in l:
      l.append(x)
  return l

# Batch detection
def batch_detection(dir_path, conf_thres = 0.55):
  all_detection = []
  dir_path = Path(dir_path)

  all_files = [fls for fls in list(dir_path.iterdir()) if fls.suffix == ".JPG"]
  pos = range(1, len(all_files) + 1)

  print(f"\nProcessing dir: {dir_path}")
  for idx, dr in enumerate(all_files):
    print(f"Image number {pos[idx]} - ({round(pos[idx]*100/len(all_files), 2)}%)")

    result = get_results(image_path = dr, conf_thres = conf_thres)

    image_id = Path(result.get('img_id')).name # image name
    cn = class_number(results = result, category='animal')
    to_save = {'image_path': f'{dr}', 'image': image_id, 'number':cn}

    all_detection.append(to_save)
    
    # Saving the batch detection results as annotated images
    batc_output = Path(dr.parent, 'detections')
    if not batc_output.exists():
      batc_output.mkdir(exist_ok=True)

    pw_utils.save_detection_images(result, output_dir=batc_output, overwrite=False)

    with open(Path(dir_path, "detection.json"), "w") as outfile: 
      json.dump(all_detection, outfile)

  return all_detection




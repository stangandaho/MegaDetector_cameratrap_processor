from detection import batch_detection
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_folder(dr):
  if dr.is_dir():
      return batch_detection(dir_path=dr)

if __name__ == "__main__":

    main_root = r"E:\DCIM_Cam4_3-2-2024"

    for cam in Path(main_root).iterdir():
        print(f"✅ON DIR: {cam}")
        species_dir = [x for x in cam.iterdir() if x.is_dir()]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            # Map the batch_detection function to each directory
            results = list(executor.map(process_folder, species_dir))
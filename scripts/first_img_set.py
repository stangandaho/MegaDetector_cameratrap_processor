from detection import batch_detection
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def process_folder(dir):
  if dir.is_dir():
      return batch_detection(dir)

if __name__ == "__main__":
    main_root = r"F:\DONNEES COLLECTEES AU NIVIEAU DES CAMERAS"
    #root_dir = r"F:\DONNEES COLLECTEES AU NIVIEAU DES CAMERAS\CAMERA 1"

    # List all directories in the root_dir
    all_dirs = [dir for dir in Path(main_root).iterdir() if dir.is_dir()]

    for sub_dir in all_dirs:
        print(f"ON DIR: {sub_dir}")
        species_dir = [dir for dir in Path(sub_dir).iterdir() if dir.is_dir()]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            # Map the batch_detection function to each directory
            results = list(executor.map(process_folder, species_dir))



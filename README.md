### Filtering Camera Trap Data Using MegaDetector

The code presented here outlines a methodology for processing and filtering camera trap data using a detection algorithm, presumably MegaDetector via [`Pytorch Wildlife`](https://github.com/microsoft/CameraTraps). The methodology is designed to handle large volumes of images collected from multiple camera traps in a parallel and efficient manner. The steps involved are as follows:

### 1. **Data Organization:**
   - **Main Directory Structure:** The camera trap data is organized in a main directory (`main_root`) containing subdirectories corresponding to individual camera trap locations (e.g., `CAMERA 1`, `CAMERA 2`, etc.). Each of these subdirectories further contains species-specific directories that hold the actual images.
   - **Subdirectory Identification:** The script identifies all the subdirectories within the `main_root`, representing different camera trap locations, and then further drills down to identify species-specific directories within each camera location.

### 2. **Detection Process:**
   - **Batch Detection Function (`batch_detection`):** The `batch_detection` function is assumed to perform the core task of processing the images in a directory using the MegaDetector algorithm. It is responsible for detecting and filtering image containing animal.
   - **Directory Processing (`process_folder`):** A helper function `process_folder` is defined to check if a given path is a directory and, if so, to invoke the `batch_detection` function on it.

### 3. **Parallel Processing:**
   - **Use of `ProcessPoolExecutor`:** To expedite the processing of potentially large datasets, the methodology employs parallel processing using Python's `concurrent.futures.ProcessPoolExecutor`. This allows multiple species directories to be processed simultaneously, significantly reducing the overall computation time.
   - **Mapping Detection to Directories:** The `ProcessPoolExecutor` is used to map the `process_folder` function to each species directory within a camera trap location, enabling concurrent execution of the detection process.

### 4. **Execution:**
   - **Main Execution Flow:** The script iterates through each camera trap location and its associated species directories. For each species directory, the detection process is executed in parallel. Results are aggregated after all directories have been processed.

The `batch_detection` function processes all .JPG images in a specified directory. It iterates through each image, performs detection, and saves the results (including the number of detected animals) in a list. Detected objects are annotated on the images, which are saved in a detections folder within the directory. Finally, the detection data is saved as a detection.json file, usable for futher data processing.

![Overview](https://github.com/stangandaho/select_animal_image/blob/main/dirview.jpg)
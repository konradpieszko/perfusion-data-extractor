from PIL import Image
import easyocr
import pandas as pd
import pydicom
from PIL import Image
import os
import random
import string
import numpy as np

# Initialize EasyOCR reader for English
reader = easyocr.Reader(['en'], gpu=True)

def extract_endo_epi_flows(image):
    """
    Given the image in the known, fixed format, crop out the 16
    second-column table cells and return endo, epi values for each row.
    
    Returns a list of dictionaries like:
       [
         {'segment': 1, 'endo': 1.98, 'epi': 2.46},
         {'segment': 2, 'endo': 1.63, 'epi': 1.92},
         ...
         {'segment': 16, 'endo': 1.88, 'epi': 1.82},
       ]
    """

    # Hard-coded bounding boxes for the 16 data rows in column 2
    SECOND_COLUMN_BOXES = [
        (520, 105 + 30*(i-1), 700, 135 + 30*(i-1))
        for i in range(3, 19)
    ]

    # Directory to save cropped images
    debug_dir = '/Users/konradpieszko/Library/CloudStorage/OneDrive-Personal/Projects_onedrive/DataExtractor dicomSR and perfu/debug_images'
    os.makedirs(debug_dir, exist_ok=True)

    # Create a unique sub-folder for this image
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    image_folder = os.path.join(debug_dir, f"image_{random_id}")
    os.makedirs(image_folder, exist_ok=True)

    # Convert to RGB (EasyOCR works with NumPy arrays)
    image = image.convert("RGB")
    
    #Save RGB image
    master_image_path = os.path.join(image_folder, "master_image_RGB.png")
    #image.save(master_image_path)

    results = []

    for i, box in enumerate(SECOND_COLUMN_BOXES, start=1):
        # Crop cell i
        cropped = image.crop(box)
        
        # Save the cropped image with a unique filename
        cropped_filename = f"cropped_image_{i}"
        cropped_path = os.path.join(image_folder, cropped_filename)
        #cropped.save(cropped_path)

        # Convert cropped image to NumPy array for easyOCR
        cropped_np = np.array(cropped)
        
        # OCR: get text from the cropped image using easyOCR
        text_list = reader.readtext(cropped_np, detail=0, paragraph=False)
        text = " ".join(text_list).strip().replace(',', '.')

        
        # Split on slash; we expect e.g. "1.98/2.46/0.80"
        parts = text.split("/")

        # Parse the first two as floats => endo, epi
        try:
            endo = float(parts[0])
            epi  = float(parts[1])
        except (ValueError, IndexError):
            endo = None
            epi = None
            print(f"Error parsing text '{text}' for segment {i}")

            results.append({"segment": i, "endo": endo, "epi": epi})
            cropped.save(f"{cropped_path}.png")
        
    return results


import pandas as pd

def convert_to_long_format(data):
    """
    Converts the list of dictionaries (segment, endo, epi) into a long-format pandas DataFrame.
    
    Args:
        data (list of dict): Output from extract_endo_epi_flows function.
    
    Returns:
        pandas.DataFrame: Long-format DataFrame with columns ['segment', 'type', 'value'].
    """
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Convert to long format using pandas melt
    long_df = df.melt(id_vars=['segment'], 
                      value_vars=['endo', 'epi'], 
                      var_name='type', 
                      value_name='value')
    
    return long_df





def dicom_series_to_dataframe(folder_path: str) -> pd.DataFrame:
    """
    Recursively scans 'folder_path' for DICOM files. 
    Returns a DataFrame with one row per frame within each DICOM series.
    
    Columns:
      - patient_name
      - patient_id
      - series_description
      - number_of_images_in_series  (total frames in the entire series)
      - path_to_image               (path to the DICOM file)
      - frame_index                 (the frame index within the DICOM file)
    """
    
    # Dictionary keyed by SeriesInstanceUID
    # series_dict[series_uid] = {
    #   'patient_name': str,
    #   'patient_id': str,
    #   'series_description': str,
    #   'file_infos': list of (file_path, number_of_frames, frame_index)
    # }
    series_dict = {}

    # Traverse all files under the given folder_path
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            fpath = os.path.join(root, fname)

            # Try reading the file as a DICOM
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except:
                # If reading fails, skip this file
                continue

            # Extract relevant metadata
            patient_name = str(ds.get("PatientName", ""))
            patient_id = str(ds.get("PatientID", ""))
            series_desc = str(ds.get("SeriesDescription", ""))
            series_uid = str(ds.get("SeriesInstanceUID", ""))
            frame_index = int(ds.get((0x0020, 0x0013), 1).value)
            
            # Debug:
            #if(series_desc=="qPerf_MBF_AI_rest__GT_FIL_Flow_Map_AHA_FIG_AI"):
                #print(f"Frame index: {frame_index} for file: {fpath}")
            # Check how many frames (multi-frame DICOM?)
            # If NumberOfFrames is absent or 0, assume 1
           
            number_of_frames = int(ds.get("NumberOfFrames", 1))
            
            # Initialize the series if not already present
            if series_uid not in series_dict:
                series_dict[series_uid] = {
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "series_description": series_desc.strip().lower()
,
                    #"frame_index": frame_index,
                    "file_infos": []
                }

            # Store the file info
            series_dict[series_uid]["file_infos"].append((fpath, number_of_frames, frame_index))

    # Build the final DataFrame rows
    rows = []
    for series_uid, info in series_dict.items():
    # Sum up the total frames for this series (across all files)
        total_frames_in_series = sum(finfo[1] for finfo in info["file_infos"])

        # We'll maintain a running index to number frames consecutively across the whole series
        frame_counter = 0

        for (fpath, n_frames,frame_index ) in info["file_infos"]:
            for _ in range(n_frames):
                rows.append({
                    "patient_name": info["patient_name"],
                    "patient_id": info["patient_id"],
                    "series_description": info["series_description"],
                    "number_of_images_in_series": total_frames_in_series,
                    "path_to_image": fpath,
                    "frame_index": frame_index
                })
                frame_counter += 1

    df = pd.DataFrame(rows)
    return df




def process_perf_series(
    df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of DICOM files (from 'dicom_series_to_dataframe'),
    finds for each patient the series:
      - "qPerf_MBF_AI_rest__GT_FIL_Flow_Map_AHA_FIG_AI"
      - "qPerf_MBF_AI_stress__GT_FIL_Flow_Map_AHA_FIG_AI"
    Converts each DICOM in those series to a PIL Image (in-memory),
    extracts endo/epi flows, converts to long format, and concatenates.

    Returns a DataFrame with columns:
      - patient_id
      - phase (rest/stress)
      - type (end/epi)
      - segment (1–16)
      - value (numeric)
    """

    # The specific series names to search for
    REST_SERIES_NAME = "qperf_mbf_ai_rest__gt_fil_flow_map_aha_fig_ai"
    STRESS_SERIES_NAME = "qperf_mbf_ai_stress__gt_fil_flow_map_aha_fig_ai"

    # We will collect all per-patient results here
    all_results = []
    
    # Group the DataFrame by patient to process each patient separately
    for patient_id, patient_df in df.groupby("patient_id"):
        #patient_df.to_csv('patient_df.csv')
        # Identify the rows belonging to the 'rest' and 'stress' series
        rest_rows = patient_df[(patient_df["series_description"] == REST_SERIES_NAME) & (patient_df["frame_index"] == 2)]
        stress_rows = patient_df[(patient_df["series_description"] == STRESS_SERIES_NAME) & (patient_df["frame_index"] == 2)]
        #patient_df[(patient_df["series_description"] == STRESS_SERIES_NAME)]
        print(f"\n Patient ID: {patient_id}, patient_name: {patient_df['patient_name'].values[0]}")
        print(stress_rows)
        print(stress_rows)
        print("End of patient \n\n\n\n")
        # A small helper to process a set of DICOM rows
        def process_phase(phase_df: pd.DataFrame, phase_name: str):
            phase_results = []
            for _, row in phase_df.iterrows():
                dcm_path = row["path_to_image"]
                # Read the DICOM and convert to PIL Image
                ds = pydicom.dcmread(dcm_path)
                
                try:
                    pixel_array = ds.pixel_array  # NumPy arrayqu
                    
                except:
                    print(f"Error converting pixel array to image for file: {dcm_path}")
                    continue

                pil_image = Image.fromarray(pixel_array)

                # Extract flows directly from the in-memory image
                table_of_flows = extract_endo_epi_flows(pil_image)

                # Convert flows to long format
                long_flows = convert_to_long_format(table_of_flows)
                # 'long_flows' expected to have columns: [segment, type, value]

                # Add patient_id and phase columns
                long_flows["patient_id"] = patient_id
                long_flows["patient_name"] = patient_df["patient_name"].values[0]
                long_flows["phase"] = phase_name

                # Reorder columns if desired
                # final columns: patient_id, phase, type, segment, value
                long_flows = long_flows[["patient_name","patient_id", "phase", "type", "segment", "value"]]

                phase_results.append(long_flows)

            if len(phase_results) > 0:
                return pd.concat(phase_results, ignore_index=True)
            else:
                return pd.DataFrame(columns=["patient_name","patient_id","phase","type","segment","value"])

        # Process rest/stress phases
        rest_result_df = process_phase(rest_rows, "rest")
        stress_result_df = process_phase(stress_rows, "stress")

        # Concatenate results for this patient
        if not rest_result_df.empty:
            all_results.append(rest_result_df)
        if not stress_result_df.empty:
            all_results.append(stress_result_df)

    # Combine results for all patients
    if len(all_results) > 0:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["patient_name","patient_id", "phase", "type", "segment", "value"])

    return final_df

list_dcm=dicom_series_to_dataframe('/Users/konradpieszko/other_data/Perf Test extraction/Casos Perfu Quant')
res=process_perf_series(list_dcm)
res.to_csv('res_all_7.csv')

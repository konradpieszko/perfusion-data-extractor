from PIL import ImageOps, ImageFilter, ImageEnhance, Image
import easyocr
import pandas as pd
import pydicom
import os
import random
import string
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib as mpl
mpl.use('Agg')

# Configurable OCR engine: "pytesseract" (default) or "easyocr"
OCR_ENGINE = "pytesseract"
#OCR_ENGINE = "easyocr"

# For pytesseract, define custom config to constrain allowed characters
CUSTOM_CONFIG = r'--psm 7 -c tessedit_char_whitelist=0123456789.'

# Initialize EasyOCR reader for English if using easyocr
if OCR_ENGINE == "easyocr":
    reader = easyocr.Reader(['en'], gpu=True)

# Set parameters for cropping within each cell
top_max = 4
bottom_min = 22

# Define new bounding boxes for the entire flow string.
# For endo: expects a string like '1.23' (from x=34 to x=61)
# For epi: expects a string like '1.23' (from x=77 to x=114)
bbox_endo = (34, top_max, 72, bottom_min)
bbox_epi = (77, top_max, 113, bottom_min)

def preprocess_digit_image(digit_region, threshold=128):
    """
    Preprocess the digit image:
      - Convert to grayscale (if necessary)
      - Perform auto-contrast enhancement
      - Apply a median filter to reduce noise
      - Apply thresholding (binarization)
    """
    # Convert to grayscale if not already
    digit = digit_region.convert("L")
    # Increase contrast automatically
    digit = ImageOps.autocontrast(digit)
    # Optionally blur to reduce noise artifacts
    digit = digit.filter(ImageFilter.MedianFilter(size=3))
    # Apply thresholding: pixels > threshold become white, else black
    digit = digit.point(lambda p: 255 if p > threshold else 0, mode='1')
    return digit

def extract_endo_epi_flows(image):
    """
    Given the image in the known, fixed format, crop out the 16
    second-column table cells and then within each cell crop out 2 regions:
      one for the endo flow and one for the epi flow.
    Each region is expected to contain a string such as '1.23'.
    OCR is performed on each region.
    
    Returns a list of dictionaries like:
       [
         {'segment': 1, 'endo': 1.23, 'epi': 2.46},
         {'segment': 2, 'endo': 1.63, 'epi': 1.92},
         ...
         {'segment': 16, 'endo': 1.88, 'epi': 1.82},
       ]
       
    If OCR does not return a properly formatted string, the function prints
    the OCR output for debugging and returns None for that value.
    Also saves the problematic cropped image for further inspection.
    """
    num_errors = 0

    # Hard-coded bounding boxes for the 16 data rows in column 2
    SECOND_COLUMN_BOXES = [
        (520, 105 + 30*(i-1), 700, 135 + 30*(i-1))
        for i in range(3, 19)
    ]

    # Directory to save cropped images for debugging
    debug_dir = '/Users/konradpieszko/Library/CloudStorage/OneDrive-Personal/Projects_onedrive/DataExtractor dicomSR and perfu/debug_images'
    os.makedirs(debug_dir, exist_ok=True)

    # Create a unique sub-folder for this image
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    image_folder = os.path.join(debug_dir, f"image_{random_id}")
    os.makedirs(image_folder, exist_ok=True)

    # Convert to greyscale and enhance contrast
    image = image.convert("RGB")
   

    results = []

    # Process each of the 16 cells
    for i, cell_box in enumerate(SECOND_COLUMN_BOXES, start=1):
        # Crop the cell from the master image
        cell_cropped = image.crop(cell_box)

        # Crop out the regions for endo and epi flow reading
        endo_region = cell_cropped.crop(bbox_endo)
        epi_region  = cell_cropped.crop(bbox_epi)

        # Preprocess both regions
        #endo_preprocessed = preprocess_digit_image(endo_region)
        #epi_preprocessed  = preprocess_digit_image(epi_region)
        endo_preprocessed = endo_region
        epi_preprocessed = epi_region

        # Function to perform OCR on a preprocessed image region
        def ocr_flow(region_img, flow_type):
            if OCR_ENGINE == "pytesseract":
                txt = pytesseract.image_to_string(region_img, config=CUSTOM_CONFIG).strip()
                if len(txt) == 5 and txt.endswith("."): #common issue that pytesseract sees dot at end
                    txt = txt[:-1]
            elif OCR_ENGINE == "easyocr":
                digit_np = np.array(region_img)
                txt_list = reader.readtext(digit_np, detail=0, paragraph=False)
                txt = " ".join(txt_list).strip()
            else:
                txt = ""
                print(f"Unknown OCR engine: {OCR_ENGINE}")
            return txt

        # Perform OCR for each flow
        ocr_endo = ocr_flow(endo_preprocessed, "endo")
        ocr_epi = ocr_flow(epi_preprocessed, "epi")

        # Basic validation: Expect a string matching pattern digit.dot+2digits (e.g., '1.23')
        def validate_flow(txt, flow_type):
            if len(txt) != 4 or not (txt[0].isdigit() and txt[1] == '.' and txt[2].isdigit() and txt[3].isdigit()):
                print(f"Error parsing {flow_type} for segment {i}. Recognised text: '{txt}'")
                
                # Save the problematic image for debugging
                txt_filename = f"segment_{i}_{flow_type}.png"
                txt_path = os.path.join(image_folder, txt_filename)
                if flow_type == "endo":
                    endo_region.save(txt_path)
                else:
                    epi_region.save(txt_path)
                return None
            return txt

        validated_endo = validate_flow(ocr_endo, "endo")
        validated_epi = validate_flow(ocr_epi, "epi")

        # Convert to float if valid, else set as None
        try:
            endo_val = float(validated_endo) if validated_endo is not None else None
        except ValueError:
            endo_val = None
        try:
            epi_val = float(validated_epi) if validated_epi is not None else None
        except ValueError:
            epi_val = None

        results.append({"segment": i, "endo": endo_val, "epi": epi_val})
        #print(f"Finished with error count: {num_errors}")
    
    return results


import pandas as pd
from PIL import ImageEnhance

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
            #patient_name = str(ds.get("PatientName", ""))
            patient_name = str(ds.get((0x0010, 0x0010), 1).value)
            patient_id = str(ds.get("PatientID", ""))
            if patient_id == "":
                patient_id = patient_name
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
    print(df)
    return df


def plot_bullseye(data, cmap='plasma', color_list=None, ax=None, title="Bull's-Eye Flow Map", 
                  show_legend=False, label_fontsize=11, fig_size=(6, 6), label_threshold=1.2,vmin=0.1, vmax=5):
    """
    Draws a bull's-eye plot for 16 segments.
    
    The input data can be:
      - a pandas DataFrame or dict with columns/keys 'segment' and 'value'
      - a pandas Series or a 1D array/list with 16 numeric values (assumed in segment order)
    
    Parameters:
      data: Input data.
      cmap: (str or Colormap) Matplotlib colormap name to map values to colors.
            Ignored if color_list is provided.
      color_list: Optional list of 16 colors (one per segment). If provided, these are used directly.
      ax: Matplotlib Axes object. If None, one is created.
      title: Title of the plot.
      show_legend: Whether to display a color legend.
      label_fontsize: Font size for segment labels.
      fig_size: Figure size if ax is not provided.
      label_threshold: Numeric threshold. Labels for segments with value lower than this will be white; otherwise black.
    """
    # Determine values and segments based on data type
    if hasattr(data, 'sort_values'):  # Likely a DataFrame
        try:
            data = data.sort_values(by='segment')
            segments = data['segment'].to_numpy()
            values = data['value'].to_numpy(dtype=float)
        except Exception as e:
            raise ValueError("DataFrame must contain 'segment' and 'value' columns.") from e
    elif isinstance(data, dict):
        try:
            segments = np.array(data['segment'])
            values = np.array(data['value'], dtype=float)
        except Exception as e:
            raise ValueError("Dictionary data must contain keys 'segment' and 'value'.") from e
    elif hasattr(data, 'to_numpy'):  # Likely a Series
        values = data.to_numpy(dtype=float)
        segments = np.arange(1, 17)  # assume segments 1..16 in order
    elif isinstance(data, (list, np.ndarray)):
        arr = np.array(data, dtype=float)
        if arr.shape[0] != 16:
            raise ValueError("Data array must have exactly 16 values.")
        values = arr
        segments = np.arange(1, 17)  # assume segments 1..16 in order
    else:
        raise ValueError("Unsupported type for data. Provide a DataFrame, dict, Series, or 1D array with 16 values.")

    if len(segments) != 16 or len(values) != 16:
        raise ValueError("Data must have exactly 16 segment values.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # Set equal aspect and full circle limits to avoid quadrant issues
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Determine color mapping if color_list not provided
    if color_list is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = mpl.cm.get_cmap(cmap)
        colors = [colormap(norm(val)) for val in values]
    else:
        if len(color_list) != 16:
            raise ValueError("color_list must contain exactly 16 colors.")
        colors = color_list

    # Define ring radii
    outer_radius = 1.0        # outer edge of basal ring
    mid_inner_radius = 0.66   # inner edge of basal = outer edge of mid
    apical_inner_radius = 0.33  # inner edge of mid = outer edge of apex

    # Angles in degrees for each wedge
    angle_per_basal = 360 / 6
    angle_per_mid   = 360 / 6
    angle_per_apical = 360 / 4

    segment_counter = 0

    # Draw basal segments (segments 1-6)
    for i in range(6):
        theta1 = i * angle_per_basal
        theta2 = theta1 + angle_per_basal
        wedge = mpl.patches.Wedge(center=(0, 0), r=outer_radius, theta1=theta1, theta2=theta2, 
                                  width=(outer_radius - mid_inner_radius), facecolor=colors[segment_counter], edgecolor='black')
        ax.add_patch(wedge)
        theta_text = np.deg2rad((theta1 + theta2) / 2)
        r_text = mid_inner_radius + (outer_radius - mid_inner_radius) / 2
        text_color = "white" if values[segment_counter] < label_threshold else "black"
        ax.text(r_text * np.cos(theta_text), r_text * np.sin(theta_text), 
                f"{values[segment_counter]:.2f}", ha='center', va='center', fontsize=label_fontsize, color=text_color)
        segment_counter += 1

    # Draw mid segments (segments 7-12)
    for i in range(6):
        theta1 = i * angle_per_mid
        theta2 = theta1 + angle_per_mid
        wedge = mpl.patches.Wedge(center=(0, 0), r=mid_inner_radius, theta1=theta1, theta2=theta2, 
                                  width=(mid_inner_radius - apical_inner_radius), facecolor=colors[segment_counter], edgecolor='black')
        ax.add_patch(wedge)
        theta_text = np.deg2rad((theta1 + theta2) / 2)
        r_text = apical_inner_radius + (mid_inner_radius - apical_inner_radius) / 2
        text_color = "white" if values[segment_counter] < label_threshold else "black"
        ax.text(r_text * np.cos(theta_text), r_text * np.sin(theta_text), 
                f"{values[segment_counter]:.2f}", ha='center', va='center', fontsize=label_fontsize, color=text_color)
        segment_counter += 1

    # Draw apical segments (segments 13-16)
    for i in range(4):
        theta1 = i * angle_per_apical
        theta2 = theta1 + angle_per_apical
        wedge = mpl.patches.Wedge(center=(0, 0), r=apical_inner_radius, theta1=theta1, theta2=theta2, 
                                  width=apical_inner_radius, facecolor=colors[segment_counter], edgecolor='black')
        ax.add_patch(wedge)
        theta_text = np.deg2rad((theta1 + theta2) / 2)
        r_text = apical_inner_radius / 2
        text_color = "white" if values[segment_counter] < label_threshold else "black"
        ax.text(r_text * np.cos(theta_text), r_text * np.sin(theta_text), 
                f"{values[segment_counter]:.2f}", ha='center', va='center', fontsize=label_fontsize, color=text_color)
        segment_counter += 1

    ax.set_title(title, fontsize=label_fontsize + 2)

    # Optionally add a color legend
    if show_legend and color_list is None:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

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
        #print(f"\n Patient ID: {patient_id}, patient_name: {patient_df['patient_name'].values[0]}")
        #print(stress_rows)
        #print(stress_rows)
        #print("End of patient \n\n\n\n")
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
        print("Processing rest phase for patient: ", patient_id)
        rest_result_df = process_phase(rest_rows, "rest")
        print("Processing stress phase for patient: ", patient_id)
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

#list_dcm=dicom_series_to_dataframe('/Users/konradpieszko/other_data/Perf Test extraction/Casos Perfu Quant')
#res=process_perf_series(list_dcm)
###res.to_csv('res_all_7.csv')
#pat_5_rest_epi= res[0:16]
#print(pat_5_rest_epi)
#plot_bullseye(pat_5_rest_epi)




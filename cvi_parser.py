import csv
import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd

# ---------------------------------------------------------
# 1) Segment-Tabular Storage Class
# ---------------------------------------------------------
class SegmentTable:
    """
    A simple class that holds a Pandas DataFrame for tabular segment-based data.
    Can be extended with methods for advanced manipulations.
    """
    def __init__(self, name="unnamed", columns: Optional[List[str]] = None):
        # Initialize an empty DataFrame or with known columns
        self.df = pd.DataFrame(columns=columns if columns else [])
        self.name = name

    def add_row(self, row_data: Dict[str, Any]):
        """
        Add a row (dict) to the internal DataFrame.
        row_data keys should match columns or new columns will be created automatically.
        """
        # Convert the row_data dict to a DataFrame with a single row
        new_row = pd.DataFrame([row_data])
        # Concatenate the new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def to_csv(self, filepath: str):
        """Export the internal DataFrame to CSV."""
        self.df.to_csv(filepath, index=False)

    def __repr__(self):
        # For debugging/printing
        return f"<Tabular data for {self.name} (SegmentTable shape={self.df.shape})>"

# ---------------------------------------------------------
# 2) Helper: Flattening the Hierarchical Data
# ---------------------------------------------------------
def flatten_hierarchy(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary. If a value is another dictionary,
    we descend into it. If a value is a SegmentTable, we store a placeholder,
    or we can produce some aggregated or stringified output depending on your needs.
    
    For the sake of having a single CSV row, we often store the presence of a
    table as either:
       - aggregated stats (e.g. "mean", "count"), or
       - a custom string marker.

    Currently, we’ll store a short text or aggregated count. Adjust as you prefer.
    """
    items = {}
    for k, v in data.items():
        # Create new key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recurse
            items.update(flatten_hierarchy(v, new_key, sep=sep))
        elif isinstance(v, SegmentTable):
            # For demonstration, let's store the shape of the DataFrame
            # or you might store something else like "SegmentTable with N rows"
            shape_str = f"SegmentTable rows={v.df.shape[0]}"
            items[new_key] = shape_str
            # Alternatively, you could do advanced flattening of each row,
            # but that often leads to many columns or multiple rows in CSV.
        else:
            # scalar or string => store directly
            items[new_key] = v
    return items

# ---------------------------------------------------------
# 3) Example Parsers
# ---------------------------------------------------------

def parse_metadata(lines, start_idx):
    """
    Parses metadata information from a specific format starting at a given index.

    Args:
        lines (list of str): The list of lines containing metadata.
        start_idx (int): The starting index from which to begin parsing.

    Returns:
        dict: A dictionary containing the parsed metadata.
    """
    metadata={}
    
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith('cvi42 Version')or line.startswith('------------------------------------------------------------------------------'):
            break

        # Simple example: "Patient    perfu_quant_test"
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        if len(parts) == 2:
            key, val = parts
            # store in metadata
            safe_key = key.lower().replace(' ', '_')
            metadata[safe_key] = val

        idx += 1
        
    return metadata, idx



def parse_quantitative_perf(lines, start_idx):
    """
    Example parser for 'Quantitative Perfusion' section.
    Returns a dictionary with metadata, plus possibly
    some segment table if needed.
    """
    metadata = {}
    result = {
        "metadata": {},  # store key-value pairs
        "coronary_teritory_averages":{}
    }
    
    idx = start_idx +1 # Add 1 to omit the ----- line after section name
    while idx < len(lines):
        
        line = lines[idx].strip()
        if line.startswith('------------------------------------------------------------------------------'):
            break
       
        # First try to read metadata, assuming it always starts with Patient:
        if line.startswith('Patient'):
            metadata, idx = parse_metadata( lines, idx)
            result['metadata']=metadata



        # Second, try to read coronary teritory averages
        if (line.startswith('Rest (MBF, ml/g/min)') or
            line.startswith('Rest (rMBF)') or
            line.startswith('rMPR') or
            line.startswith('Stress (rMBF)') or
            line.startswith('MPR')):


            result['coronary_teritory_averages'][line], idx = parse_coronary_territory_averages(lines, idx, line)

        idx += 1

    return result, idx

import pandas as pd

def aggregate_tabular_data(data_dict):
    """
    Aggregates multiple SegmentTable instances into two Pandas DataFrames:
      1) aggregated_segment_df
      2) aggregated_artery_df
    
    Each row corresponds to the name attribute of the SegmentTable
    Each column corresponds to a segment or artery territory
    An 'average' column is appended to each DataFrame.
    """
    # Storage for rows
    segment_rows = []
    artery_rows = []

    # Grab the dictionary containing the segment and artery data
    territory_dict = data_dict['quantitative_perf']['coronary_teritory_averages']

    # Loop over each measurement key (e.g. "Rest (MBF, ml/g/min)", "MPR", etc.)
    for measure_name, measure_data in territory_dict.items():
        # --- Segment data ---
        seg_table = measure_data['segment_data'] # This is a SegmentTable
      
        # Set segment as index, select 'value', and transpose
        seg_wide = seg_table.df.set_index('segment')['value'].to_frame().T
        # Now seg_wide is a 1×16 table:
        #   columns = unique segments
        #   single row containing the mean 'value' for each segment

    
       
        # Label this row by the table's name attribute
        seg_wide.index = [seg_table.name]  # e.g. "Rest (MBF, ml/g/min)"
        print(seg_wide)
        # Add average column
        seg_wide = seg_wide.apply(pd.to_numeric, errors='coerce', axis=0)
        seg_wide['average'] = seg_wide.mean(axis=1)
        segment_rows.append(seg_wide)

        # --- Artery data ---
        art_table = measure_data['artery_data']  # Another SegmentTable for artery data
        # Pivot in the same way (assuming "segment" column in df is actually artery territory)
     
        art_wide = art_table.df.set_index('artery')['value'].to_frame().T
        art_wide.index = [art_table.name]
        # Add average column
        art_wide = art_wide.apply(pd.to_numeric, errors='coerce', axis=0)
        art_wide['average'] = art_wide.mean(axis=1)
        artery_rows.append(art_wide)

    # Concatenate all the wide rows into a single DataFrame each
    aggregated_segment_df = pd.concat(segment_rows, axis=0)
    aggregated_artery_df = pd.concat(artery_rows, axis=0)

    return aggregated_segment_df, aggregated_artery_df


def parse_coronary_territory_averages(lines, start_idx, name):
    """
    Example parser for the block with segment-based data.
    We'll demonstrate how to create and fill a SegmentTable.
    e.g. lines like:
        Segment 1   1.6929
        Segment 2   1.4706
        ...
    """
    # We might store the data in a SegmentTable. We'll define columns: "segment" and "value".
    seg_table = SegmentTable(name=name, columns=["segment", "value"])
    territory_table= SegmentTable(name=name, columns=["artery", "value"])

    idx = start_idx +1
    while idx < len(lines):
        line = lines[idx].strip()
    

        
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        if len(parts) >= 2:
            label, val_str = parts
            # e.g. "Segment 1", "LAD Territory Average", ...
            # we only store segments in the table for example
            if label.lower().startswith('segment'):
                # extract segment number
                # label might be "Segment 1", let's parse
                seg_num_str = label.split()[1]  # "1"
                row_data = {
                    "segment": seg_num_str,
                    "value": val_str
                }
                seg_table.add_row(row_data)
            elif line:
                row_data = {
                    "artery": label,
                    "value": val_str
                }
                territory_table.add_row(row_data)
    

        idx += 1
        if(line.startswith('Segment 16')): # always finish after segment 16
           break

    # wrap in a dictionary
    data = {"segment_data": seg_table, "artery_data": territory_table  }
    return data, idx


def parse_file(filepath: str) -> Dict[str, Any]:
    """
    Parse an entire file and return a hierarchical data structure.
    """
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    # We'll create a top-level dict where each section is a key
    # (like "quantitative_perf", "coronary_territory", etc.)
    all_data = {
        "patient_info": {},
        "quantitative_perf": {},
        "coronary_territory_averages": {},
        # ... add more top-level keys as needed
    }

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        if "Quantitative Perfusion" in line:
            # parse
            res, new_idx = parse_quantitative_perf(lines, idx + 1)
            all_data["quantitative_perf"].update(res)
            seg_df, artery_df = aggregate_tabular_data(all_data)
            all_data["aggregated_segment_perfusion_data"] = seg_df
            all_data["aggregated_territory_perfusion_data"] = artery_df

            idx = new_idx
            continue

      

        else:
            idx += 1

    return all_data

# ---------------------------------------------------------
# 4) Flatten + Export to CSV
# ---------------------------------------------------------
def flatten_and_write_csv(data_list: List[Dict[str, Any]], out_csv: str):
    """
    data_list is a list of hierarchical dictionaries (one per file).
    1) Flatten each dictionary
    2) Collect all columns
    3) Write CSV
    """
    print("flattening")
    flattened_data = []
    for d in data_list:
        flat_d = flatten_hierarchy(d)
        flattened_data.append(flat_d)

    # Gather all columns
    all_keys = set()
    for fd in flattened_data:
        all_keys.update(fd.keys())
    all_keys = sorted(all_keys)

    with open(out_csv, 'w', newline='', encoding='utf-16') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for fd in flattened_data:
            writer.writerow(fd)

# ---------------------------------------------------------
# Example: parse multiple files
# ---------------------------------------------------------
def parse_multiple_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    results = []
    for fp in filepaths:
        if os.path.isfile(fp):
            hierarchical_data = parse_file(fp)
            # you might also store the filename in hierarchical_data
            hierarchical_data["file_info"] = {"source_filename": os.path.basename(fp)}
            print(hierarchical_data)
            results.append(hierarchical_data)
        else:
            print(f"File not found: {fp}")
    return results

# ---------------------------------------------------------
# If you want to export a specific SegmentTable from your hierarchy
# individually, you can navigate to it, e.g.:
# all_data["coronary_territory_averages"]["rest_mbf_table"].to_csv("rest_mbf_segments.csv")
# or implement a specialized function to find all SegmentTable objects.
# ---------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    file_list = ["example_txt_reports/perfu_quant_test_2024-12-05_Scientific_Report_2024-12-27_full.txt"]#, "example_txt_reports/perfu_quant_test_2024-12-05_Scientific_Report_2024-12-27_2.txt"]
    data_list = parse_multiple_files(file_list)
    #print(data_list)
    # Flatten + write a single CSV (one row per file)
    flatten_and_write_csv(data_list, "combined.csv")
    print("\n\n")
    # If you want to export a particular DataFrame from the first file’s structure:
    #print(data_list[0])
    #print(data_list[0]['quantitative_perf']['coronary_teritory_averages']['Rest (MBF, ml/g/min)']['segment_data'].df)
    data_list[0]["aggregated_segment_perfusion_data"].to_csv("aggregated_segment_data.csvy")
    data_list[0]["aggregated_territory_perfusion_data"].to_csv("aggregated_territory.csv")
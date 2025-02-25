import os
import tempfile
import zipfile
import shutil
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, redirect, url_for

# Import your functions from your existing module
from gadgetron_ocr_quant_data_extractor_v3 import dicom_series_to_dataframe, process_perf_series, plot_bullseye
from gould_plot import create_cfr_stressflow_plot

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save the uploaded ZIP file to a temporary directory and extract it.
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "upload.zip")
        file.save(zip_path)
        extract_dir = os.path.join(tmp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Process the DICOM series
        df_dicom = dicom_series_to_dataframe(extract_dir)
        df_perf = process_perf_series(df_dicom)

        # Iterate over patients and pick the first one with both phases and both types present (with 16 segments each)
        patient_id = None
        patient_name = None
        table_html = None
        bullseye_data = {}
        for pid, group in df_perf.groupby("patient_id"):
            # Make sure patient_name is available (assuming same for all rows)
            name = group["patient_name"].iloc[0] if "patient_name" in group.columns else "Unknown"

            # Check that the patient has both phases and both flow types with 16 segments each
            valid = True
            temp = {}
            for phase in ["rest", "stress"]:
                for ftype in ["endo", "epi"]:
                    sub = group[(group["phase"] == phase) & (group["type"] == ftype)]
                    if len(sub) != 16:
                        valid = False
                        break
                    # Sort according to segment order
                    sub_sorted = sub.sort_values("segment")
                    # Store the 16 flow values; assume 'flow' is the column 'value'
                    temp[f"{phase}_{ftype}"] = sub_sorted["value"].values
                if not valid:
                    break

            if valid:
                patient_id = pid
                patient_name = name
                bullseye_data = temp
                break

        shutil.rmtree(tmp_dir)

        if patient_id is None:
            return "No valid patient data found with complete 16-segment series for rest and stress for both endo and epi.", 400

        # Compute total flows per segment as weighted average: 0.6 for endo and 0.4 for epi
        rest_total = 0.6 * bullseye_data["rest_endo"] + 0.4 * bullseye_data["rest_epi"]
        stress_total = 0.6 * bullseye_data["stress_endo"] + 0.4 * bullseye_data["stress_epi"]

        # Compute MPR (stress/rest) per segment for each type and total
        mpr_endo = bullseye_data["stress_endo"] / bullseye_data["rest_endo"]
        mpr_epi = bullseye_data["stress_epi"] / bullseye_data["rest_epi"]
        mpr_total = stress_total / rest_total

        # Create a summary table with mean flows (without patient name and id)
        

        summary = {
            "Rest Endo Flow": bullseye_data["rest_endo"],
            "Rest Epi Flow": bullseye_data["rest_epi"],
            "Rest Total Flow": rest_total,
            "Stress Endo Flow": bullseye_data["stress_endo"],
            "Stress Epi Flow": bullseye_data["stress_epi"],
            "Stress Total Flow": stress_total,
            "MPR Endo": mpr_endo,
            "MPR Epi": mpr_epi,
            "MPR Total": mpr_total
        }
        table_df = pd.DataFrame(summary)
        table_html = table_df.to_html(classes="table table-striped", index=False, float_format=lambda x: f"{x:.2f}")

        # Create figure with 3x3 bullseye plots.
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        # Mapping of rows and columns:
        # Rows: 0->Total, 1->Endo, 2->Epi
        # Cols: 0->Rest, 1->Stress, 2->MPR
        # Prepare data for each subplot
        data_dict = {
            (0, 0): (rest_total, "Rest Total Flow"),
            (0, 1): (stress_total, "Stress Total Flow"),
            (0, 2): (mpr_total, "MPR Total"),
            (1, 0): (bullseye_data["rest_endo"], "Rest Endo Flow"),
            (1, 1): (bullseye_data["stress_endo"], "Stress Endo Flow"),
            (1, 2): (mpr_endo, "MPR Endo"),
            (2, 0): (bullseye_data["rest_epi"], "Rest Epi Flow"),
            (2, 1): (bullseye_data["stress_epi"], "Stress Epi Flow"),
            (2, 2): (mpr_epi, "MPR Epi")
        }
        for (r, c), (data, title) in data_dict.items():
            ax = axes[r, c]
            # Pass the colormap arguments to plot_bullseye.
            plot_bullseye(data, ax=ax, cmap="plasma", vmin=0.1, vmax=5)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Adjust layout to allow room for a single colorbar on the right.
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        norm = plt.Normalize(vmin=0.1, vmax=5)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)


        # Create CFR vs Stress Flow plot
        points = np.column_stack((stress_total, mpr_total))
        buf2 = BytesIO()
        create_cfr_stressflow_plot(points, save_path=None, file_format='png', show_plot=True, poly_alpha=1)
        plt.savefig(buf2, format="png", bbox_inches="tight")
        buf2.seek(0)
        cfr_plot_img = base64.b64encode(buf2.getvalue()).decode("utf-8")


        return render_template("index.html", table=table_html, plot_img=plot_img, patient_name=patient_name, patient_id=patient_id, gould_plot=cfr_plot_img)

    return render_template("index.html", table=None, plot_img=None, patient_name=None, patient_id=None)

if __name__ == "__main__":
    app.run(debug=True)

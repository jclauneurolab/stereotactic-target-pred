from flask import jsonify, request, send_file, after_this_request

from utils import app, logger, log_tracebook, bad_request, server_error

import os
import yaml
from apply_model import model_pred
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
print(root_dir)

configfile = os.path.join(root_dir, "config", "config.yaml")

with open(configfile, "r") as file:
    config = yaml.safe_load(file)

UPLOAD_FOLDER = os.path.join(root_dir, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/apply-model", methods=["POST"])
def post_model_predict_json():
    """
    Apply the model and predict the stereotaxy coordinates.

    Attributes
    ----------

    Returns
    -------
        Flask JSON response
    """
    try:
        try:            
            if not os.access(UPLOAD_FOLDER, os.W_OK):
                print("Warning: No write permission for UPLOAD_FOLDER")

            file = request.files.get("file")

            model_type = request.form.get("model_type")
            logger.info("model_type: %s", model_type)
            
        except Exception as e:
            log_tracebook(e)
            logger.error(
                "Inputs to predict steoreotaxy coordinates" + 
                "do not match the match required inputs"
            )

            return bad_request(
                "Inputs to predict stereotaxy coordinates" + 
                "do not match the required inputs."
            )

        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        logger.info("Saving file to: %s", file_path)
        file.save(file_path)

        file_name_without_extension = os.path.splitext(os.path.basename(file.filename))[0]
        logger.info("No extension: %s", file_name_without_extension)

        OUTPUT_FOLDER = os.path.join(root_dir,"output", f"{file_name_without_extension}_output")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Run the model prediction
        slicer_tfm = f'{OUTPUT_FOLDER}/{file_name_without_extension}_ACPC.txt'
        logger.info("slicer_tfm: %s", slicer_tfm)

        template_fcsv = os.path.join(root_dir, config.get("template_fcsv"))
        logger.info("template_fscv: %s", template_fcsv)

        midpoint = 'PMJ'
        logger.info("midpoint: %s", midpoint)

        model_path = os.path.join(root_dir, config.get(model_type))
        logger.info("model_path: %s", model_path)
        
        target_mcp = f'{OUTPUT_FOLDER}/{file_name_without_extension}_mcp.fcsv'
        logger.info("target_mcp: %s", target_mcp)

        target_native = f'{OUTPUT_FOLDER}/{file_name_without_extension}_native.fcsv'
        logger.info("target_native: %s", target_native)

        print("------------")
        print("starting model pred")
        print("------------")

        model_pred(
            in_fcsv=file_path,
            model=model_path,
            midpoint=midpoint,
            slicer_tfm=slicer_tfm,
            template_fcsv=template_fcsv,
            target_mcp=target_mcp,
            target_native=target_native
        )
        return jsonify({"message": f"Model ran"}), 200

    except Exception as e:
        log_tracebook(e)
        logger.error(
            "There was a problem predicting stereotaxy coordinates " + "and returning some of its attributes in a JSON format."
        )
        return server_error(
            "There was a problem predicting stereotaxy coordinates " + "and returning some of its attributes in a JSON format."
        )

@app.route("/download-output", methods=["GET"])
def get_download_output_json():
    """
    Download the output with the stereotaxy predicted results

    Attributes
    ----------

    Returns
    -------
        send_file method
    """
    try:
        try:
            file_name_without_extension = request.args.get("file_name")
        except Exception as e:
            log_tracebook(e)
            logger.error(
                "Inputs to download the predicted stereotaxy coordinates " + "do not match required inputs."
            )

            return bad_request(
                "Inputs to download the predicted stereotaxy coordinates " + "do not match the required inputs."
            )
        
        output_folder = os.path.join(root_dir, "output", f"{file_name_without_extension}_output")
        
        output_zip_path = os.path.join(root_dir, f"{file_name_without_extension}_output.zip")
        
        # Zip the output folder into a .zip file
        shutil.make_archive(output_zip_path.replace(".zip", ""), 'zip', output_folder)

        @after_this_request
        def remove_file(response):
            try:
                os.remove(output_zip_path)
                print(f"Deleted ZIP file: {output_zip_path}")
            except Exception as e:
                print(f"Error deleting file {output_zip_path}: {e}")
            return response
        
        # Send the ZIP file to the frontend
        return send_file(output_zip_path, as_attachment=True, download_name=f"{file_name_without_extension}_output.zip")
    
    except Exception as e:
        log_tracebook(e)
        logger.error(
            "There was a problem downloading the predicted stereotaxy coordinates " + "and returning some of its attributes in a JSON format."
        )

        return server_error(
            "There was a problem downloading the predicted stereotaxy coordinates " + "and returning some of its attributes in a JSON format."
        )

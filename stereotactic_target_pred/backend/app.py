from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import yaml
from apply_model import model_pred
import shutil

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
print(root_dir)

configfile = os.path.join(root_dir, "config", "config.yaml")

with open(configfile, "r") as file:
    config = yaml.safe_load(file)

UPLOAD_FOLDER = os.path.join(root_dir, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OUTPUT_FOLDER = os.path.join(root_dir, "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("starting pred")

@app.route("/apply-model", methods=["POST"])
def predict():
    print("Request received")
    print("Request headers:", request.headers)
    
    if not os.access(UPLOAD_FOLDER, os.W_OK):
        print("Warning: No write permission for UPLOAD_FOLDER")

    file = request.files.get("file")
    model_type = request.form.get("model_type")
    print(model_type)

    if not file or not model_type:
        return jsonify({"error": "Missing file or model type"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    print("Saving file to:", file_path)
    file.save(file_path)

    # Run the model prediction
    try:
        slicer_tfm = f'{OUTPUT_FOLDER}/{file.filename}_ACPC.txt'
        print(slicer_tfm)
        template_fcsv = os.path.join(root_dir, config.get("template_fcsv"))
        print(template_fcsv)
        midpoint = 'PMJ'
        print(midpoint)
        model_path = os.path.join(root_dir, config.get(model_type))
        print(model_path)
        target_mcp = f'{OUTPUT_FOLDER}/{file.filename}_mcp.fcsv'
        print(target_mcp)
        target_native = f'{OUTPUT_FOLDER}/{file.filename}_native.fcsv'
        print(target_native)

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
        return jsonify({"message": f"Error processing the file: {str(e)}"}), 500


@app.route("/download-output", methods=["GET"])
def download_output():
    output_zip_path = os.path.join(root_dir, "{file.filename}_output.zip")

    # Zip the output folder
    shutil.make_archive(output_zip_path.replace(".zip", ""), 'zip', OUTPUT_FOLDER)

    # Send the ZIP file to the frontend
    return send_file(output_zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

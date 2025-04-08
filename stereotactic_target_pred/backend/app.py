from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
from flask_cors import CORS
import os
import yaml
from apply_model import model_pred
import shutil

app = Flask(__name__, static_folder="../frontend/dist", static_url_path='/')
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
    file_name_without_extension = os.path.splitext(os.path.basename(file.filename))[0]
    print("no extension", file_name_without_extension)

    OUTPUT_FOLDER = os.path.join(root_dir,"output", f"{file_name_without_extension}_output")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Run the model prediction
    try:
        slicer_tfm = f'{OUTPUT_FOLDER}/{file_name_without_extension}_ACPC.txt'
        print(slicer_tfm)
        template_fcsv = os.path.join(root_dir, config.get("template_fcsv"))
        print(template_fcsv)
        midpoint = 'PMJ'
        print(midpoint)
        model_path = os.path.join(root_dir, config.get(model_type))
        print(model_path)
        target_mcp = f'{OUTPUT_FOLDER}/{file_name_without_extension}_mcp.fcsv'
        print(target_mcp)
        target_native = f'{OUTPUT_FOLDER}/{file_name_without_extension}_native.fcsv'
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
    file_name_without_extension = request.args.get("file_name")
    if not file_name_without_extension:
        return jsonify({"error": "File name is required"}), 400
    
    output_folder = os.path.join(root_dir, "output", f"{file_name_without_extension}_output")
    
    if not os.path.exists(output_folder):
        return jsonify({"error": "Output folder not found"}), 404
    
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

# Serve the React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)

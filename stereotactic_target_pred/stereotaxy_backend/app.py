from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import yaml
from apply_model import model_pred

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

configfile = '/Users/mackenziesnyder/Desktop/stereotactic-target-pred/config/config.yaml'
with open(configfile, "r") as file:
    config = yaml.safe_load(file)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OUTPUT_FOLDER = "./output"
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

    if not file or not model_type:
        return jsonify({"error": "Missing file or model type"}), 400

    # if not file.filename.endswith(".fcsv"):
    #     return jsonify({"error": "Only .fcsv files are allowed"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    print("Saving file to:", file_path)
    file.save(file_path)

    # Run the model prediction
    try:
        # You can replace these with your actual paths and arguments
        slicer_tfm = f'{OUTPUT_FOLDER}/ACPC.txt'
        print(slicer_tfm)
        template_fcsv = config["template_fcsv"]
        print(template_fcsv)
        midpoint = 'PMJ'
        print(midpoint)
        model_path = config["{model_type}"]
        print(model_path)
        target_mcp = f'{OUTPUT_FOLDER}/mcp.fcsv'
        print(target_mcp)
        target_native = f'{OUTPUT_FOLDER}/native.fcsv'
        print(target_native)

        # Call the prediction function (model_pred)
        zip_path = model_pred(
            in_fcsv=file_path,
            model_path=model_path,
            midpoint=midpoint,
            slicer_tfm=slicer_tfm,
            template_fcsv=template_fcsv,
            target_mcp=target_mcp,
            target_native=target_native
        )

        return jsonify({"message": f"Results saved at: {zip_path}"}), 200

    except Exception as e:
        return jsonify({"message": f"Error processing the file: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

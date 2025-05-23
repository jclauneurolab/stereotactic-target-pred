import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import '../stylesheets/file_upload.css';
import { API_URL } from '../env'

const FileUpload = () => {
  const [selectedModel, setSelectedModel] = useState("");
  const [file, setFile] = useState(null);
  const [success, setSuccess] = useState(false);
  const [fileNameWithoutExtension, setFileNameWithoutExtension] = useState("");

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: ".fcsv",
    onDrop: (acceptedFiles) => {
      const uploadedFile = acceptedFiles[0];
      setFile(uploadedFile);
      const fileNameWithoutExt = uploadedFile.name.split('.')[0];
      setFileNameWithoutExtension(fileNameWithoutExt);
    },
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !selectedModel) {
      alert("Please select a file and a model type!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_type", selectedModel);
    console.log("file", { file }, "model_type", { selectedModel });

    try {
      const response = await axios.post(`${API_URL}/apply-model`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert(`Success: ${response.data.message}`);
      setSuccess(true);
    } catch (error) {
      console.error("Upload failed:", error.response || error);
      alert("Upload failed!");
      setSuccess(false);
    }
  };

  const handleDownload = async () => {
    if (!fileNameWithoutExtension) {
      alert("No file to download");
      return;
    }

    try {
      const response = await axios.get(`${API_URL}/download-output?file_name=${fileNameWithoutExtension}`, {
        responseType: "blob",
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `${fileNameWithoutExtension}_output.zip`);
      document.body.appendChild(link);
      link.click();
    } catch (error) {
      console.error("Download failed:", error);
      alert("Download failed!");
    }
  };

  return (
    <div>
      <p className="autoafids-content">
        To automatically produce the anatomical fiducial .fcsv file from a NIFTI MRI image, please refer to the{" "}
        <a href="https://github.com/afids/autoafids" target="_blank" rel="noopener noreferrer">
          AutoAFIDs program
        </a>.
      </p>

            <div className="protocol-note">
            <p><strong>Note:</strong> Ensure your uploaded file abides by these specifications:</p>
            <ul>
              <li>File name starts with <code>sub-"label"</code></li>
              <li>The label column within the <code>.fscv</code> file abides by the AFIDs labels (e.g., <code>AC = 1</code>) shown in the Protocol tab</li>
              <li>Upload not working? Check out this demo template <a href="/public/tpl-MNI2009cAsym_afids.fcsv" download>here</a></li>
            </ul>
            </div>
      <div className="container">
        <div {...getRootProps()} className="dropzone">
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the file here...</p>
          ) : (
            <p>Drag & drop an AFIDs .fcsv file here, or click to select</p>
          )}
        </div>
        {file && <p>Selected File: {file.name}</p>}
        <select onChange={(e) => setSelectedModel(e.target.value)}>
          <option value="">Select Prediction Model</option>
          <option value="STN">STN</option>
          <option value="cZI">cZI(not supported yet)</option>
          <option value="cZI">CM(not supported yet)</option>

        </select>
        <button onClick={handleSubmit}>Submit</button>
        {success && (
          <div>
            <button onClick={handleDownload}>Download Output</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;

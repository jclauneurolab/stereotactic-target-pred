import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import '../stylesheets/file_upload.css';

const FileUpload = () => {
  const [selectedModel, setSelectedModel] = useState(""); 
  const [file, setFile] = useState(null);
  const [success, setSuccess] = useState(false);
  const [visuals, setVisuals] = useState(null);
  const [showTargetDropdown, setShowTargetDropdown] = useState(false);
  const [targetType, setTargetType] = useState(""); 
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
    console.log("file", {file}, "model_type", {selectedModel});

    try {
      const response = await axios.post("http://127.0.0.1:5001/apply-model", formData, {
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
      const response = await axios.get(`http://127.0.0.1:5001/download-output?file_name=${fileNameWithoutExtension}`, {
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

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "output.zip");
    document.body.appendChild(link);
    link.click();
  }

  const showVisuals = async () => {
    if (visuals) {
      setVisuals(null); 
      setShowTargetDropdown(false);
      return;
    }
    
    try {
       const formData = new FormData();
       formData.append("file", file);
   
       const response = await axios.post("http://127.0.0.1:5000/visualizations", formData, {
         headers: { "Content-Type": "multipart/form-data" }
       });
 
       console.log("should show visuals here")
       console.log("Response Data:", response.data);
   
       setVisuals(response.data.scatter); 
       setShowTargetDropdown(true);
       console.log("dropdown should be showing")
     } catch (error) {
       console.error("Error fetching visualization:", error);
       alert("Failed to load visualization.");
     }
   };
 
  const updateVisualizationWithTarget = async () => {  
    if (!file || !targetType) {
      alert("Please select a target type and ensure a file is uploaded!");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file); 
    formData.append("targetType", targetType);
    
    console.log("target is:", targetType);
    console.log("file is:", file.name);
  
    try {
      const response = await axios.post("http://127.0.0.1:5000/visualizations", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setVisuals(response.data.scatter);

    } catch (error) {
      alert("Upload failed!");
      console.error("Error updating visualization:", error);
    }
  };

  return (
    <div>
      <div className="container">
        <div {...getRootProps()} className="dropzone">
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the file here...</p>
          ) : (
            <p>Drag & drop an afids .fcsv file here, or click to select</p>
          )}
        </div>
        {file && <p>Selected File: {file.name}</p>}
        <select onChange={(e) => setSelectedModel(e.target.value)}>
          <option value="">Select Model</option>
          <option value="STN">STN</option>
          <option value="cZI">cZI</option>
        </select>
        <button onClick={handleSubmit}>Upload</button>
        {success && (
          <div className="output-buttons">
            <button onClick={handleDownload}>Download Output</button>
            <button onClick={showVisuals}>Show Visualizations</button>
          </div>
        )}
        {showTargetDropdown && (
         <div className="output-buttons">
           <select onChange={(e) => setTargetType(e.target.value)}>
             <option value="">Select Target Type</option>
             <option value="native">Native</option>
             <option value="mcp">MCP</option>
           </select>
           <button onClick={updateVisualizationWithTarget}>Upload</button>
         </div>
       )}
       {visuals && (
          <div className="graph">
            <iframe
              srcDoc={visuals}
              title="3D Scatter Plot"
              style={{ width: "100%", height: "500px", border: "none" }}
            />
          </div>
       )}
      </div>
    </div>
  );
};

export default FileUpload;

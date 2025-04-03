import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";

const FileUpload = () => {
  const [selectedModel, setSelectedModel] = useState(""); 
  const [file, setFile] = useState(null);
  const [success, setSuccess] = useState(false);
  // const [visuals, setVisuals] = useState(null);
  // const [showTargetDropdown, setShowTargetDropdown] = useState(false);
  // const [targetType, setTargetType] = useState(""); 

  // Handle file drop
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: ".fcsv",
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !selectedModel) {
      alert("Please select a file and a model type!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_type", selectedModel);
    console.log("file", {file}, "model_type", {selectedModel})

    try {
      const response = await axios.post("http://127.0.0.1:5000/apply-model", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert(`Success: ${response.data.message}`);
      setSuccess(true);
    } catch (error) {
      alert("Upload failed!");
      console.error(error);
      setSuccess(false);
    }
  };

  const handleDownload = async () => {
    const response = await axios.get("http://127.0.0.1:5000/download-output", {
      responseType: "blob", // Important for file downloads
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "output.zip");
    document.body.appendChild(link);
    link.click();
  }

  // const showVisuals = async () => {
  //   try {
  //     const formData = new FormData();
  //     formData.append("file", file);
  
  //     const response = await axios.post("http://127.0.0.1:5000/visualizations", formData, {
  //       headers: { "Content-Type": "multipart/form-data" }
  //     });

  //     console.log("should show visuals here")
  //     console.log("Response Data:", response.data);
  
  //     setVisuals(response.data.scatter); 
  //     // setShowTargetDropdown(true);
  //   } catch (error) {
  //     console.error("Error fetching visualization:", error);
  //     alert("Failed to load visualization.");
  //   }
  // };

  // const updateVisualizationWithTarget = async () => {  
    
  //   const formData = new FormData();
  //   formData.append("targetType", targetType);

  //   try {
  //     const response = await axios.post("http://127.0.0.1:5000/visualizations", formData, {
  //       headers: { "Content-Type": "multipart/form-data" },
  //     });
  //     alert(`Success: ${response.data.message}`);
  //     setVisuals(response.data.scatter);
  //   } catch (error) {
  //     alert("Upload failed!");
  //     console.error(error);
  //   }
  // };
  
  return (
    <div className="container">
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the file here...</p>
        ) : (
          <p>Drag & drop an .fcsv file here, or click to select</p>
        )}
      </div>
      <select onChange={(e) => setSelectedModel(e.target.value)}>
        <option value="">Select Model</option>
        <option value="STN">STN</option>
        <option value="cZI">cZI</option>
      </select>
      {file && <p>Selected File: {file.name}</p>}
      <button onClick={handleSubmit}>Upload</button>
      {success && (
        <div>
          <button onClick={handleDownload}>Download Output</button>
          {/* <button onClick={showVisuals}>Show Visualizations</button> */}
        </div>
      )}

      {/* {showTargetDropdown && (
        <div>
          <select onChange={(e) => setTargetType(e.target.value)}>
            <option value="">Select Target Type</option>
            <option value="native">Native</option>
            <option value="mcp">MCP</option>
          </select>
          <button onClick={updateVisualizationWithTarget}>Upload</button>
        </div>
      )}

      {/* Render 3D Plot if Available */}
      {/* {visuals && (
        <iframe
          srcDoc={visuals}
          title="3D Scatter Plot"
          style={{ width: "100%", height: "500px", border: "none" }}
        />
      )} */}
    </div>
  );
};

export default FileUpload;

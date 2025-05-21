import React, { useState } from "react";
import './App.css';
import FileUpload from "./components/file_upload";
import Navbar from "./components/navbar";

function App() {
  const [activeTab, setActiveTab] = useState("upload");

  return (
    <div>
      <Navbar />

      {/* Tab Buttons */}
      <div className="tabs">
        <button onClick={() => setActiveTab("upload")}>Upload</button>
        <button onClick={() => setActiveTab("protocol")}>Protocol</button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === "upload" && <FileUpload />}

        {activeTab === "protocol" && (
          <div>
            <h2>AFIDs Protocol (16 AFIDs)</h2>
            {activeTab === "protocol" && (
          <div>
            <div className="protocol-note">
              <p>
                <strong>Note:</strong> This page shows a <em>subset</em> of 16 anatomical fiducials from the full AFIDs protocol for illustrative purposes.
                If this is your first time manually annotating, we strongly recommend reviewing the complete protocol, which includes essential details on:
                setting up <strong>3D Slicer</strong>, performing the <strong>AC-PC transform</strong>, and adhering to the proper <strong>.fcsv naming conventions</strong>.
                These steps are crucial to ensure accurate landmark placement and model predictions.
              </p>
              <p>
                You can find the full protocol here:{" "}
                <a href="https://afids.github.io/afids-protocol/" target="_blank" rel="noopener noreferrer">
                  AFIDs Protocol Repository
                </a>
              </p>
            </div>

            {landmarkData.map((lm) => (
              <div className="landmark" key={lm.id}>
                <h3>{lm.title}</h3>
                <p><strong>Description:</strong> {lm.desc}</p>
                <img
                  src={`/protocol_imgs/${lm.id}.png`}
                  alt={`${lm.title} image`}
                  style={{ maxWidth: "600px", width: "100%", margin: "1rem auto" }}
                />
              </div>
            ))}
          </div>
        )}

          {landmarkData.map((lm) => (
            <div className="landmark" key={lm.id}>
              <h3>{lm.title}</h3>
              <p><strong>Description:</strong> {lm.desc}</p>
              <img
                src={`/protocol_imgs/${lm.id}.png`}
                alt={`${lm.title} image`}
                style={{ maxWidth: "600px", width: "100%", margin: "1rem auto" }}
              />
            </div>
          ))}
        </div>
      )}
      </div>
    </div>
  );
}
const landmarkData = [
  { id: "01_AC", title: "1. AC [midline]", desc: "Place at the center of the anterior commissure." },
  { id: "02_PC", title: "2. PC [midline]", desc: "Place at the center of the posterior commissure." },
  { id: "03_ICS", title: "3. Infracollicular Sulcus [midline]", desc: "Inferior boundary of the intercollicular sulcus at the junction of the inferior colliculi." },
  { id: "04_PMJ", title: "4. Pontomesencephalic Junction [midline]", desc: "Select the inferior/pontine edge of the PMJ using sagittal and coronal views." },
  { id: "05_SIF", title: "5. Superior Interpeduncular Fossa [midline]", desc: "Dorsal aspect of the interpeduncular fossa between the cerebral peduncles." },
  { id: "06_RSLMS", title: "6. Right Superior Lateral Mesencephalic Sulcus", desc: "Right superior angle of the brainstem at the mesencephalic surface." },
  { id: "07_LSLMS", title: "7. Left Superior Lateral Mesencephalic Sulcus", desc: "Left superior angle of the brainstem at the mesencephalic surface." },
  { id: "08_RILMS", title: "8. Right Inferior Lateral Mesencephalic Sulcus", desc: "Junction between midbrain and pons on the right side." },
  { id: "09_LILMS", title: "9. Left Inferior Lateral Mesencephalic Sulcus", desc: "Junction between midbrain and pons on the left side." },
  { id: "11_IMS", title: "11. Intermammillary Sulcus [midline]", desc: "Midpoint between the mammillary bodies, placed at the grey matter boundary." },
  { id: "12_RMB", title: "12. Right Mammillary Body", desc: "Center of the right mammillary body." },
  { id: "13_LMB", title: "13. Left Mammillary Body", desc: "Center of the left mammillary body." },
  { id: "15_RLVAC", title: "15. Right Lateral Ventricle at AC", desc: "Lateral aspect of the frontal horn at the AC level (right side)." },
  { id: "16_LLVAC", title: "16. Left Lateral Ventricle at AC", desc: "Lateral aspect of the frontal horn at the AC level (left side)." },
  { id: "17_RLVPC", title: "17. Right Lateral Ventricle at PC", desc: "Lateral aspect of the frontal horn at the PC level (right side)." },
  { id: "18_LLVPC", title: "18. Left Lateral Ventricle at PC", desc: "Lateral aspect of the frontal horn at the PC level (left side)." }
];


export default App;

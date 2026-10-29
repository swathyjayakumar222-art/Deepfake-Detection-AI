import { useState } from "react";
import {
  AlertTriangle,
  LoaderCircle,
  RotateCcw,
  ScanFace,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function chooseFile(nextFile) {
    setError("");
    setResult(null);
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError("Image must be 10 MB or smaller.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(API_URL + "/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not analyze the image.");
      setResult(data);
    } catch (err) {
      setError(
        (err.message || "Request failed") +
          ". Ensure the API is running and a trained model checkpoint exists."
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  }

  const fakePercent = result ? (result.fake_probability * 100).toFixed(1) : "0.0";
  const realPercent = result ? (result.real_probability * 100).toFixed(1) : "0.0";

  return (
    <main className="page-shell">
      <nav className="nav">
        <div className="brand">
          <span className="brand-mark"><ShieldCheck size={21} /></span>
          <span>Deepfake<span className="brand-light">Detect</span></span>
        </div>
        <span className="status"><span className="status-dot" /> ML LAB · PROTOTYPE</span>
      </nav>

      <section className="hero">
        <div className="eyebrow"><span /> MEDIA AUTHENTICITY RESEARCH</div>
        <h1>See beyond the <span>synthetic.</span></h1>
        <p className="hero-copy">
          Analyze a face image with a machine-learning model trained on real and
          manipulated examples. Treat the result as a signal—not a verdict.
        </p>
      </section>

      <section className="workspace">
        <div className="panel">
          <div className="panel-heading">
            <div><span className="step">01</span><h2>Upload an image</h2></div>
            <ScanFace size={21} />
          </div>
          <label
            className={"dropzone " + (dragging ? "dragging " : "") + (preview ? "has-preview" : "")}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            {preview ? (
              <img className="preview" src={preview} alt="Selected image preview" />
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon"><UploadCloud size={25} /></span>
                <strong>Drop an image here</strong>
                <span>or click to browse your files</span>
                <small>JPG · PNG · WEBP · Max 10 MB</small>
              </div>
            )}
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
          </label>
          {file && (
            <div className="file-row">
              <span>{file.name}</span>
              <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
            </div>
          )}
          <button className="primary-button" disabled={!file || loading} onClick={analyze}>
            {loading ? <><LoaderCircle className="spin" size={18} /> Analyzing…</> :
              <><ScanFace size={18} /> Analyze image</>}
          </button>
          {file && <button className="text-button" onClick={reset}><RotateCcw size={14} /> Clear image</button>}
          {error && <div className="error-box"><AlertTriangle size={17} /><span>{error}</span></div>}
        </div>

        <div className="panel result-panel">
          <div className="panel-heading">
            <div><span className="step">02</span><h2>Model assessment</h2></div>
            <span className="live-label">INFERENCE</span>
          </div>
          {!result ? (
            <div className="empty-result">
              <div className="radar"><div /><ScanFace size={31} /></div>
              <strong>Awaiting analysis</strong>
              <p>The model prediction and class probabilities will appear here after inference.</p>
              <div className="placeholder-line" /><div className="placeholder-line short" />
            </div>
          ) : (
            <div className="result-content">
              <div className={"result-symbol " + result.prediction}><ShieldCheck size={30} /></div>
              <span className="result-kicker">MODEL PREDICTION</span>
              <h3 className={result.prediction}>
                {result.prediction === "fake" ? "Potentially manipulated" : "Classified as real"}
              </h3>
              <p className="result-note">
                This is a learned pattern classification, not proof that media is authentic or fake.
              </p>
              <div className="probability">
                <div className="prob-label"><span>Fake probability</span><b>{fakePercent}%</b></div>
                <div className="track"><span style={{ width: fakePercent + "%" }} /></div>
              </div>
              <div className="probability">
                <div className="prob-label"><span>Real probability</span><b>{realPercent}%</b></div>
                <div className="track real-track"><span style={{ width: realPercent + "%" }} /></div>
              </div>
              <div className="result-footer"><span>EfficientNet-B0</span><span>Threshold: 0.50</span></div>
            </div>
          )}
        </div>
      </section>

      <section className="disclaimer">
        <AlertTriangle size={18} />
        <p><strong>Research prototype.</strong> This model can make mistakes and may fail on manipulation
          methods or image sources not represented in training. Do not use its output alone to accuse
          a person or make high-stakes decisions.</p>
      </section>
      <footer>
        <span>DEEPFAKE DETECTION AI</span>
        <span>Built for learning · Train locally, evaluate responsibly</span>
      </footer>
    </main>
  );
}

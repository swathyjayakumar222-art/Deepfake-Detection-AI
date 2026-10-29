# Deepfake Detection AI

An educational deep-learning project that trains a binary image classifier to distinguish real and manipulated face images. It includes a PyTorch training pipeline, evaluation metrics, a FastAPI inference service, and a React upload dashboard.

> **Research prototype only.** Predictions can be wrong; performance on one dataset does not guarantee generalization to unseen manipulation methods. It is not forensic proof.

## Train in Google Colab (recommended)

[**Open the training notebook in Google Colab**](https://colab.research.google.com/github/swathyjayakumar222-art/Deepfake-Detection-AI/blob/main/notebooks/Train_Deepfake_Detector_Colab.ipynb)

1. Enable a GPU in Colab via **Runtime → Change runtime type → T4 GPU**, if available.
2. Prepare a labeled image dataset in your Google Drive using the structure below.
3. Run the notebook cells in order. It fine-tunes EfficientNet-B0, evaluates the held-out test split, and saves `best_model.pt` and `metrics.json` back to your Drive.

The notebook and code are committed to GitHub; **the model is not trained yet** because no dataset has been supplied and no training run has completed.

## Dataset setup

The dataset is deliberately not included in the repository. Use data you are allowed to access. One research option is [FaceForensics++](https://github.com/ondyari/FaceForensics), whose maintainers require an access request and acceptance of their terms. Follow the dataset’s license and terms.

The training code expects images organized like this:

```text
data/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

Each class folder must contain images. If you start with videos, extract usable face frames and split the source videos/identities into train, validation, and test **before** extracting frames. Never put frames from the same source video in more than one split, as that can inflate test scores through data leakage.

## Train locally

Python 3.10–3.12 is recommended. A GPU helps; CPU training is possible but slower.

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --data-dir ../data --epochs 5 --batch-size 16
```

The best validation-loss checkpoint is saved to `backend/artifacts/best_model.pt`; held-out test metrics are written to `backend/artifacts/metrics.json`. Adjust epochs and batch size to suit your hardware.

## Run the API

```bash
cd backend
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`. The API returns HTTP 503 until a trained checkpoint exists.

## Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. Set `VITE_API_URL` if the API is hosted somewhere other than `http://127.0.0.1:8000`.

## Evaluation checklist

- Keep the test set untouched until model selection is complete.
- Report precision, recall, F1, ROC-AUC and the confusion matrix—not accuracy alone.
- Inspect false positives/negatives and class balance.
- Test on a different source or dataset if feasible.
- Document the dataset, split strategy, hardware, hyperparameters and random seed.
- Do not claim that the model detects every deepfake.

## Planned research extensions

1. Compare EfficientNet-B0 against a custom CNN.
2. Add video frame extraction and video-level prediction aggregation.
3. Add Grad-CAM explanations, clearly labeled as approximate.
4. Evaluate cross-dataset generalization and probability calibration.

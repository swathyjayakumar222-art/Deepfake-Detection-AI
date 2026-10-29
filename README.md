# Deepfake Detection AI

An educational deep-learning project that trains a binary image classifier to distinguish real and manipulated face images. It includes a reproducible PyTorch training pipeline, evaluation metrics, a FastAPI inference service, and a React frontend.

> Research prototype only. Predictions can be wrong; performance on one dataset does not guarantee generalization to unseen manipulation methods.

## Dataset setup

Use a dataset you are permitted to access, such as FaceForensics++ (follow its access and license terms). Organize a labeled subset as:

```text
data/
  train/{fake,real}/
  val/{fake,real}/
  test/{fake,real}/
```

Prevent leakage by splitting by source video/identity before extracting frames. Do not put near-duplicate frames from the same video across train and test. The dataset is not included in this repository.

## Train

Python 3.10–3.12 is recommended. A GPU helps; CPU training is possible but slower.

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --data-dir ../data --epochs 5 --batch-size 16
```

A best checkpoint is saved to `backend/artifacts/best_model.pt`; evaluation results go to `backend/artifacts/metrics.json`. Start small and adjust batch size/epochs to available compute.

## Run API

```bash
cd backend
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`. The API returns HTTP 503 until you train a checkpoint.

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the API is not running at `http://127.0.0.1:8000`.

## Evaluation

- Keep the test set untouched until model selection is complete.
- Report precision, recall, F1 and ROC-AUC, not only accuracy.
- Inspect false positives/negatives and class balance.
- Test on a different source/dataset if feasible.
- Document split strategy, dataset provenance, hardware, hyperparameters and seed.
- Never claim the model detects every deepfake.

## Planned improvements

1. Compare against a small custom CNN.
2. Add video frame extraction and prediction aggregation.
3. Add Grad-CAM explanations, clearly labeled as approximate.
4. Evaluate cross-dataset generalization and calibration.

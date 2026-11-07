# Skin Lesion Classifier — Production Package

A deployable version of the HAM10000 skin-lesion classifier: same modeling ideas as the
assignment notebook (ConvNeXt-Tiny, and a ConvNeXt+Transformer hybrid), but restructured
so it can actually be trained anywhere and shipped to other people as a working app —
not tied to Colab or a grading rubric.

## Files
- `train.py` — trains the model(s) from a local (or mounted) copy of HAM10000 and exports
  a `.keras` file + `label_map.json`. No Google Drive dependency.
- `app.py` — Streamlit web app. Loads whichever exported model(s) it finds and serves
  predictions with a simple, shareable UI.
- `requirements.txt`, `Dockerfile` — for running/deploying the app.

## 1. Train (once)
On a CPU / low-RAM machine use MobileNetV2; on a GPU you can use `hybrid` or `both`.
```bash
python train.py \
  --metadata_csv ./data/ham10000/HAM10000_metadata.csv \
  --image_dirs ./data/ham10000/HAM10000_images_part_1 ./data/ham10000/HAM10000_images_part_2 \
  --output_dir ./models \
  --architecture mobilenet \
  --epochs_head 10 --epochs_finetune 20
```
Use `--architecture both` to train ConvNeXt and the hybrid together — the app will
automatically ensemble them if both `.keras` files are present, usually a small extra
accuracy bump over either model alone.

This produces, in `./models` (copy into the app folder to deploy):
- `mobilenet_final.keras` (and/or `hybrid_final.keras` / `convnext_final.keras`)
- `label_map.json`
- `all_metrics.json` / `*_metrics.json` — accuracy/F1/ROC-AUC on the held-out test set

## 2. Run the app locally
```bash
# put the trained model file(s) + label_map.json next to app.py
cp models/*.keras models/label_map.json .
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Upload an image, get a prediction.

## 3. Ship it to other people (pick one)

### Option A — Streamlit Community Cloud (free, easiest, gives you a public URL)
1. Push this folder (including the `.keras` model file(s) and `label_map.json`) to a
   GitHub repo. If the model file is large (ConvNeXt-Tiny is a few hundred MB), use
   [Git LFS](https://git-lfs.github.com/).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the
   main file to `app.py`, deploy.
3. You get a public `https://<yourapp>.streamlit.app` link you can send to anyone.

### Option B — Hugging Face Spaces (free, handles large model files well)
1. Create a new Space → SDK: **Streamlit**.
2. Upload `app.py`, `requirements.txt`, the `.keras` file(s), and `label_map.json`
   (HF Spaces has generous file-size limits and native large-file storage).
3. It builds and gives you a public URL automatically.

### Option C — Docker, anywhere (Render, Railway, Fly.io, a VPS, your own server)
```bash
docker build -t skin-classifier .
docker run -p 8501:8501 skin-classifier
```
Push the image to any container host (Render, Railway, Fly.io, AWS/GCP/Azure, etc.)
and point it at port 8501.

## Notes on making this genuinely production-grade
- **Model format**: exports to native Keras (`.keras`), not the legacy `.h5` used in
  the notebook — this is the format TF/Keras recommends going forward and loads more
  reliably across TF versions.
- **Reproducible inference**: `label_map.json` is written once by `train.py` and read
  by `app.py`, so the class order the app displays always matches what the model was
  actually trained on — no risk of a hand-copied class list drifting out of sync.
- **Ensembling for free**: if you train both architectures, the app averages their
  predictions (plus flip-based TTA) automatically — no extra code needed to get the
  accuracy benefit of an ensemble in production.
- **Responsible framing**: the app leads with a clear "not a medical device" disclaimer.
  If you're actually distributing this beyond a demo, you'd want a real model card,
  logging/monitoring, and — if it's ever used in an actual clinical or consumer-health
  context — regulatory review (e.g. FDA guidance on AI/ML-based medical software) before
  it touches real patient decisions.
- **Scaling beyond a demo**: Streamlit is great for sharing with a handful/hundreds of
  people. For higher traffic, wrap the model in a small FastAPI service (`/predict`
  endpoint) behind a proper autoscaled container deployment, and make the current
  Streamlit app (or a lightweight frontend) call that API instead of loading the model
  in-process.

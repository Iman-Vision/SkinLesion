# Production image for the Skin Lesion Classifier Streamlit app.
# Build:  docker build -t skin-classifier .
# Run:    docker run -p 8501:8501 skin-classifier
# (Assumes mobilenet_final.keras / label_map.json are already in this
#  folder — produced by train.py)

FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow/TensorFlow, plus curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ui ./ui
COPY label_map.json .
# Copy whichever trained model file(s) exist
COPY *.keras ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

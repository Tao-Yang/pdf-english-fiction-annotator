# Web app image for the PDF English-fiction annotator (Gradio).
# Deploy to Hugging Face Spaces (Docker SDK), Render, Railway or any Docker host.
FROM python:3.11-slim

# System libs required by PyMuPDF / Pillow at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

# Copy the application code.
COPY annotator /app/annotator
COPY webapp /app/webapp

# Prebuild the disk-backed dictionary and NLTK data into the image. This avoids
# downloading a 65 MB CSV and expanding it into a huge Python dict on Render's
# memory-constrained free instances.
ENV ANNOTATOR_DATA_DIR=/app/.cache/data \
    NLTK_DATA=/app/.cache/nltk_data \
    PORT=7860
RUN mkdir -p /app/.cache/data /app/.cache/nltk_data \
    && python webapp/prepare_assets.py --data-dir /app/.cache/data \
    && python -c "from annotator.nltk_setup import ensure_nltk_data; ensure_nltk_data()" \
    && chmod -R 777 /app/.cache

EXPOSE 7860

CMD ["python", "webapp/app.py"]

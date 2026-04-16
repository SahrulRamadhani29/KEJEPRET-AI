FROM python:3.11-slim

# Tambahkan user standar HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Switch kembali ke root HANYA untuk install apt-get
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy folder model dan kode
COPY --chown=user:user models/ /app/models/
COPY --chown=user:user app.py .

# Kembalikan akses eksekusi ke user 1000
USER user

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
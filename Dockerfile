# Base image officielle Python
FROM python:3.10-slim

# Empêche les invites interactives pendant apt install
ENV DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de l'application
WORKDIR /app
COPY . .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Port par défaut pour Streamlit
EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "app.py"]

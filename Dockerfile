FROM python:3.10-slim

# Installer les outils nécessaires pour compiler des paquets
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY ./Docker/dependency /app/dependency
COPY ./rag.py /app/rag.py
COPY ./utils /app/utils
# Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/dependency/requirements.txt

# Exposer le port 80
EXPOSE 80

# Commande par défaut
CMD ["python", "rag.py"]

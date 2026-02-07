#!/bin/bash
# Avvia il backend FastAPI (porta 8000)

cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
    echo "Creazione ambiente virtuale..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f "venv/bin/uvicorn" ]; then
    echo "Installazione dipendenze..."
    pip install -r requirements.txt
fi

echo "Backend in ascolto su http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000

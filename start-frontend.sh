#!/bin/bash
# Avvia il frontend Vite (porta 5173)

cd "$(dirname "$0")/frontend"

if [ ! -d "node_modules" ]; then
    echo "Installazione dipendenze npm..."
    npm install
fi

echo "Frontend in ascolto su http://localhost:5173"
npm run dev

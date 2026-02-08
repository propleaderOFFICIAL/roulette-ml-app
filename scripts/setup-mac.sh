#!/usr/bin/env bash
# Setup per macOS: installa libomp (richiesto da XGBoost) e dipendenze backend.
# Esegui dalla root del repo: ./scripts/setup-mac.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"

echo "=== Setup roulette-ml-app su macOS ==="

# 1) Homebrew e libomp (necessario per XGBoost su Mac)
if [[ "$(uname)" != "Darwin" ]]; then
  echo "Questo script è pensato per macOS. Salto installazione libomp."
else
  if ! command -v brew &>/dev/null; then
    echo "Homebrew non trovato. Installalo da https://brew.sh"
  else
    echo "Installazione libomp (OpenMP) per XGBoost..."
    if brew install libomp 2>/dev/null; then
      echo "libomp installato."
    else
      echo "Impossibile installare libomp (es. permessi Homebrew)."
      echo "Per abilitare XGBoost su Mac esegui poi: sudo chown -R \$(whoami) /opt/homebrew && brew install libomp"
    fi
  fi
fi

# 2) Venv e dipendenze Python
cd "$BACKEND"
if [ ! -d "venv" ]; then
  echo "Creazione venv..."
  python3 -m venv venv
fi
source venv/bin/activate

echo "Installazione dipendenze Python (incluso XGBoost)..."
pip install --upgrade pip
pip install -r requirements.txt

# Verifica XGBoost
echo ""
if python3 -c "import xgboost; print('XGBoost versione:', xgboost.__version__)" 2>/dev/null; then
  echo "XGBoost installato correttamente."
else
  echo "Avviso: XGBoost non disponibile. L'app funziona comunque con gli altri modelli."
fi

echo ""
echo "Setup completato. Avvia il backend con: ./start-backend.sh"

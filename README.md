# Roulette ML App

App React + backend Python (FastAPI) per registrare le uscite della **roulette europea** (numeri 0–36) e visualizzare probabilità **teoriche**, **empiriche** (frequenze dai dati) e **previsioni** da un modello di machine learning.

## Requisiti

- **Backend**: Python 3.10+ (consigliato 3.11)
- **Frontend**: Node.js 18+ e npm

## Avvio

### 1. Backend (porta 8000)

```bash
cd roulette-ml-app/backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Oppure, dalla root del progetto:

```bash
./start-backend.sh
```

### 2. Frontend (porta 5173)

```bash
cd roulette-ml-app/frontend
npm install
npm run dev
```

Poi apri [http://localhost:5173](http://localhost:5173).

## Utilizzo

1. **Registra uscita**: inserisci un numero da 0 a 36 e clicca "Aggiungi". L’ultima uscita viene mostrata con il colore (rosso/nero/verde).
2. **Probabilità**: la sezione centrale mostra:
   - **Teoriche**: 18/37 rosso, 18/37 nero, 1/37 verde (roulette europea).
   - **Empiriche**: percentuali calcolate sulle uscite che hai registrato.
   - **Modello ML**: dopo circa 15+ uscite, viene addestrato un modello che usa le ultime 5 uscite per stimare probabilità sul prossimo colore/numero.
3. **Storico**: in basso vedi l’elenco delle ultime uscite con numero, colore e orario.

I dati delle uscite sono salvati in `backend/data/spins.json` e restano tra un avvio e l’altro del backend.

## Nota importante

Nella roulette reale ogni colpo è **indipendente**: le probabilità teoriche non cambiano in base alla storia. Le “previsioni” del modello e le frequenze empiriche sono fornite a scopo **didattico e di intrattenimento**, non come strategia per battere il banco.

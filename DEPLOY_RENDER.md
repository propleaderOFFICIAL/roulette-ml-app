# Deploy su Render

Puoi mettere **backend** e **frontend** su Render in due modi: con il Blueprint (automatico) o creando i servizi a mano.

---

## Opzione 1: Blueprint (consigliato)

1. **Push del progetto su GitHub**  
   Assicurati che il repo contenga la cartella `roulette-ml-app` (o che il root del repo sia proprio `roulette-ml-app`). Se il repo è “Progetti Antigravity” con dentro `roulette-ml-app`, in Render userai **root directory** (vedi sotto).

2. **Vai su [Render](https://render.com)** e accedi. Clicca **New** → **Blueprint**.

3. **Collega il repository** GitHub (o GitLab) e scegli il repo.  
   - Se il repo è la root “Progetti Antigravity” e `render.yaml` sta in `roulette-ml-app/render.yaml`, in **Root Directory** imposta: `roulette-ml-app`.  
   - Se il repo contiene solo l’app roulette e `render.yaml` è nella root, lascia **Root Directory** vuoto.

4. **Applica il Blueprint**  
   Render creerà due servizi:
   - **roulette-ml-api** (Web Service, backend)
   - **roulette-ml-frontend** (Static Site, frontend)

5. **Configura l’URL del backend per il frontend**  
   - Nel dashboard Render apri il servizio **roulette-ml-frontend**.
   - Vai in **Environment** e aggiungi:
     - **Key:** `VITE_API_URL`  
     - **Value:** `https://roulette-ml-api.onrender.com`  
     (sostituisci con l’URL reale del backend se Render ti ha dato un nome diverso, es. `https://roulette-ml-api-xxxx.onrender.com`).
   - Salva e avvia un **Manual Deploy** per il frontend (così il build usa la nuova variabile).

6. **Apri l’app**  
   L’interfaccia sarà su qualcosa tipo:  
   `https://roulette-ml-frontend.onrender.com`

---

## Opzione 2: Creare i servizi a mano

### Backend (Web Service)

1. **New** → **Web Service**.
2. Collega il repo e, se serve, imposta **Root Directory**: `roulette-ml-app/backend` (o `backend` se il repo è già `roulette-ml-app`).
3. Imposta:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Clicca **Create Web Service**.  
   Annotati l’URL del backend (es. `https://roulette-ml-api.onrender.com`).

### Frontend (Static Site)

1. **New** → **Static Site**.
2. Stesso repo; **Root Directory**: `roulette-ml-app/frontend` (o `frontend` se il repo è già `roulette-ml-app`).
3. Imposta:
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `dist`
4. In **Environment** aggiungi:
   - **Key:** `VITE_API_URL`
   - **Value:** l’URL del backend (es. `https://roulette-ml-api.onrender.com`)
5. **Create Static Site**.

L’app sarà disponibile all’URL del frontend che Render assegna (es. `https://roulette-ml-frontend.onrender.com`).

---

## Se vedi "404" o "Failed to load resource: 404"

Il frontend è una SPA: tutte le route devono restituire `index.html`. Nel `render.yaml` è già configurata una **rewrite** (`/*` → `/index.html`). Dopo un nuovo deploy del Blueprint la regola viene applicata.

Se il 404 persiste (es. dopo un deploy fatto prima di questa modifica), aggiungi la regola a mano:
- Dashboard → **roulette-ml-frontend** → **Redirects/Rewrites**
- **Add Rule**: Source `/*`, Destination `/index.html`, Action **Rewrite**

Se il 404 riguarda un **URL dell’API** (es. `roulette-ml-api.onrender.com/...`), controlla che il backend sia avviato (piano free: può essere in sleep; apri l’URL del backend e attendi il risveglio).

---

## Note importanti

- **Piano free:** il backend dopo ~15 minuti di inattività va in sleep; la prima richiesta dopo il sleep può richiedere 30–60 secondi (cold start).
- **Dati:** su Render il filesystem è effimero. I dati delle uscite (es. `data/spins.json`) si perdono a ogni redeploy o riavvio. Per tenerli bisognerebbe usare un database (es. Postgres su Render) o uno storage esterno.
- **CORS:** il backend accetta richieste da qualsiasi origine se `CORS_ORIGINS` non è impostata. Se il frontend non si connette, in Render → roulette-ml-api → Environment verifica che `CORS_ORIGINS` non ci sia o sia vuota. Se vuoi limitare al solo frontend, nel servizio backend imposta la variabile d’ambiente `CORS_ORIGINS` con l’URL del frontend (es. `https://roulette-ml-frontend.onrender.com`).

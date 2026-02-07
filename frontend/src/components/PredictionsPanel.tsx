import { PredictionsResponse } from '../api';

interface PredictionsPanelProps {
  data: PredictionsResponse | null;
  loading: boolean;
  error: string | null;
}

// Roulette number mappings
const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

const getColor = (n: number): string => {
  if (n === 0) return 'green';
  return RED_NUMBERS.includes(n) ? 'red' : 'black';
};

export function PredictionsPanel({ data, loading, error }: PredictionsPanelProps) {
  if (error) {
    return (
      <div className="ai-predictions-page">
        <h2 className="page-title">🤖 Predizioni AI</h2>
        <div className="error-card">
          <p className="error-message">{error}</p>
        </div>
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="ai-predictions-page loading-state">
        <h2 className="page-title">🤖 Predizioni AI</h2>
        <div className="loading-card">
          <div className="spinner"></div>
          <p>Caricamento predizioni...</p>
        </div>
      </div>
    );
  }

  const { model, betting_areas, total_spins } = data;
  const needsMoreSpins = total_spins < 30;

  // Get color prediction
  const colorPrediction = model?.color
    ? Object.entries(model.color).reduce((a, b) => (a[1] > b[1] ? a : b))
    : null;

  return (
    <div className="ai-predictions-page">
      <h2 className="page-title">
        🤖 Predizioni AI
        <span className="spin-badge">{total_spins} uscite</span>
      </h2>

      {/* Info state: need more spins */}
      {needsMoreSpins && (
        <div className="info-banner">
          <span className="info-icon">ℹ️</span>
          <div>
            <strong>Servono più dati!</strong>
            <p>Inserisci almeno {30 - total_spins} altre uscite per attivare le predizioni AI complete.</p>
          </div>
        </div>
      )}

      {/* === COLORE (Rosso/Nero) === */}
      <section className="prediction-section">
        <h3>🎨 Prossimo Colore</h3>
        {model?.color ? (
          <div className="color-prediction-main">
            <div className={`color-winner color-${colorPrediction?.[0]}`}>
              <span className="predicted-label">{colorPrediction?.[0] === 'red' ? 'ROSSO' : colorPrediction?.[0] === 'black' ? 'NERO' : 'VERDE'}</span>
              <span className="predicted-prob">{((colorPrediction?.[1] ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="color-bars-horizontal">
              {Object.entries(model.color).map(([color, prob]) => (
                <div key={color} className="color-bar-item">
                  <span className="color-name">{color === 'red' ? 'Rosso' : color === 'black' ? 'Nero' : 'Verde'}</span>
                  <div className="bar-track">
                    <div className={`bar-fill bg-${color}`} style={{ width: `${prob * 100}%` }} />
                  </div>
                  <span className="bar-value">{(prob * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="waiting-data">
            <span>⏳</span>
            <p>In attesa di dati sufficienti per la predizione colore</p>
          </div>
        )}
      </section>

      {/* === DOZZINE E COLONNE === */}
      <section className="prediction-section">
        <h3>📊 Dozzine e Colonne</h3>
        {betting_areas ? (
          <div className="betting-grid-2col">
            {/* Dozzine */}
            <div className="betting-card">
              <h4>Dozzine</h4>
              <div className="predicted-winner">
                🎯 {betting_areas.dozen.prediction}
                <span className="conf">{(betting_areas.dozen.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mini-bars">
                {Object.entries(betting_areas.dozen.probabilities).map(([name, prob]) => (
                  <div key={name} className="mini-bar-row">
                    <span className="mini-label">{name}</span>
                    <div className="mini-track"><div className="mini-fill" style={{ width: `${prob * 100}%` }} /></div>
                    <span className="mini-val">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Colonne */}
            <div className="betting-card">
              <h4>Colonne</h4>
              <div className="predicted-winner">
                🎯 {betting_areas.column.prediction}
                <span className="conf">{(betting_areas.column.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mini-bars">
                {Object.entries(betting_areas.column.probabilities).map(([name, prob]) => (
                  <div key={name} className="mini-bar-row">
                    <span className="mini-label">{name}</span>
                    <div className="mini-track"><div className="mini-fill" style={{ width: `${prob * 100}%` }} /></div>
                    <span className="mini-val">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Alto/Basso */}
            <div className="betting-card">
              <h4>Alto / Basso</h4>
              <div className="predicted-winner">
                🎯 {betting_areas.high_low.prediction}
                <span className="conf">{(betting_areas.high_low.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mini-bars">
                {Object.entries(betting_areas.high_low.probabilities).map(([name, prob]) => (
                  <div key={name} className="mini-bar-row">
                    <span className="mini-label">{name}</span>
                    <div className="mini-track"><div className="mini-fill" style={{ width: `${prob * 100}%` }} /></div>
                    <span className="mini-val">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Pari/Dispari */}
            <div className="betting-card">
              <h4>Pari / Dispari</h4>
              <div className="predicted-winner">
                🎯 {betting_areas.parity.prediction}
                <span className="conf">{(betting_areas.parity.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mini-bars">
                {Object.entries(betting_areas.parity.probabilities).map(([name, prob]) => (
                  <div key={name} className="mini-bar-row">
                    <span className="mini-label">{name}</span>
                    <div className="mini-track"><div className="mini-fill" style={{ width: `${prob * 100}%` }} /></div>
                    <span className="mini-val">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="waiting-data">
            <span>⏳</span>
            <p>Servono almeno 30 uscite per le predizioni sulle aree di scommessa</p>
          </div>
        )}
      </section>

      {/* === NUMERI SINGOLI === */}
      <section className="prediction-section">
        <h3>🔢 Numeri Singoli Più Probabili</h3>
        {model?.top_numbers && model.top_numbers.length > 0 ? (
          <div className="numbers-grid">
            {model.top_numbers.slice(0, 10).map(({ number: n, probability: p }, idx) => (
              <div key={n} className={`number-card color-${getColor(n)} ${idx === 0 ? 'top-pick' : ''}`}>
                <span className="num">{n}</span>
                <span className="pct">{(p * 100).toFixed(1)}%</span>
                {idx === 0 && <span className="top-label">TOP</span>}
              </div>
            ))}
          </div>
        ) : (
          <div className="waiting-data">
            <span>⏳</span>
            <p>In attesa di dati sufficienti per le predizioni sui numeri singoli</p>
          </div>
        )}
      </section>

      {/* Zero Warning */}
      {betting_areas && betting_areas.zero_probability > 0.05 && (
        <div className="zero-alert">
          ⚠️ Alta probabilità dello Zero: {(betting_areas.zero_probability * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

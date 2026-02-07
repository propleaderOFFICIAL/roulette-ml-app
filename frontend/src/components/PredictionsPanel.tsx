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

const getDozen = (n: number): number => {
  if (n === 0) return 0;
  if (n <= 12) return 1;
  if (n <= 24) return 2;
  return 3;
};

const getColumn = (n: number): number => {
  if (n === 0) return 0;
  return ((n - 1) % 3) + 1;
};

// Calculate betting area stats from numbers
function calculateBettingStats(numbers: Array<{ number: number }>) {
  if (numbers.length === 0) return null;

  const stats = {
    dozen: { 1: 0, 2: 0, 3: 0 },
    column: { 1: 0, 2: 0, 3: 0 },
    highLow: { alto: 0, basso: 0 },
    parity: { pari: 0, dispari: 0 },
    zeros: 0,
  };

  numbers.forEach(({ number: n }) => {
    if (n === 0) {
      stats.zeros++;
      return;
    }
    const d = getDozen(n);
    const c = getColumn(n);
    if (d >= 1 && d <= 3) stats.dozen[d as 1 | 2 | 3]++;
    if (c >= 1 && c <= 3) stats.column[c as 1 | 2 | 3]++;
    stats.highLow[n <= 18 ? 'basso' : 'alto']++;
    stats.parity[n % 2 === 0 ? 'pari' : 'dispari']++;
  });

  const total = numbers.length;
  const nonZeroTotal = total - stats.zeros;

  return {
    dozen: {
      '1ª (1-12)': stats.dozen[1] / total,
      '2ª (13-24)': stats.dozen[2] / total,
      '3ª (25-36)': stats.dozen[3] / total,
    },
    column: {
      '1ª col': stats.column[1] / total,
      '2ª col': stats.column[2] / total,
      '3ª col': stats.column[3] / total,
    },
    highLow: {
      'Basso (1-18)': nonZeroTotal > 0 ? stats.highLow.basso / nonZeroTotal : 0,
      'Alto (19-36)': nonZeroTotal > 0 ? stats.highLow.alto / nonZeroTotal : 0,
    },
    parity: {
      'Pari': nonZeroTotal > 0 ? stats.parity.pari / nonZeroTotal : 0,
      'Dispari': nonZeroTotal > 0 ? stats.parity.dispari / nonZeroTotal : 0,
    },
    zeroRate: stats.zeros / total,
  };
}

function ProbBar({ value, colorClass, label }: { value: number; colorClass: string; label?: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="prob-bar-row">
      {label && <span className="prob-label">{label}</span>}
      <div className="prob-bar-container">
        <div
          className={`prob-bar-fill ${colorClass}`}
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
      <span className="prob-value">{pct}%</span>
    </div>
  );
}

function ColorProbs({
  title,
  probs,
}: {
  title: string;
  probs: Record<string, number>;
}) {
  return (
    <div className="prob-group">
      <h4>{title}</h4>
      <div className="color-bars">
        <ProbBar value={probs.red ?? 0} colorClass="bg-red" label="Rosso" />
        <ProbBar value={probs.black ?? 0} colorClass="bg-black" label="Nero" />
        <ProbBar value={probs.green ?? 0} colorClass="bg-green" label="Verde" />
      </div>
    </div>
  );
}

export function PredictionsPanel({ data, loading, error }: PredictionsPanelProps) {
  if (error) {
    return (
      <div className="predictions-panel card">
        <h2>📊 Probabilità</h2>
        <p className="error-message">{error}</p>
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="predictions-panel card loading-state">
        <div className="spinner"></div>
        <p>Caricamento probabilità...</p>
      </div>
    );
  }

  const { theoretical, empirical, model, total_spins } = data;

  return (
    <div className="predictions-panel">
      <h2 className="section-title">
        <span className="icon">📊</span>
        Probabilità Prossima Uscita
        {total_spins > 0 && <span className="spin-count">({total_spins} uscite)</span>}
      </h2>

      {total_spins === 0 && (
        <div className="empty-state">
          <span className="empty-icon">🎰</span>
          <p>Nessuna uscita registrata</p>
          <p className="hint">Inserisci i numeri che escono alla roulette per vedere le statistiche</p>
        </div>
      )}

      {/* Color Predictions */}
      <div className="predictions-grid">
        <div className="prediction-card">
          <ColorProbs title="🎯 Teoriche (roulette europea)" probs={theoretical.color} />
          <p className="note">Singolo numero: {(theoretical.number_probability * 100).toFixed(2)}%</p>
        </div>

        {total_spins > 0 && (
          <div className="prediction-card">
            <ColorProbs title="📈 Empiriche (dai tuoi dati)" probs={empirical.color} />
            {empirical.top_numbers.length > 0 && (
              <div className="top-numbers">
                <h5>Numeri più frequenti</h5>
                <div className="number-chips">
                  {empirical.top_numbers.slice(0, 5).map(({ number: n, probability: p }) => (
                    <div key={n} className={`number-chip color-${getColor(n)}`}>
                      <span className="chip-number">{n}</span>
                      <span className="chip-prob">{(p * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {model && (model.color || (model.top_numbers && model.top_numbers.length > 0)) && (
          <div className="prediction-card highlight">
            {model.color && (
              <ColorProbs title="🤖 Previsione ML" probs={model.color} />
            )}
            {model.top_numbers && model.top_numbers.length > 0 && (
              <div className="top-numbers">
                <h5>Numeri predetti dall'AI</h5>
                <div className="number-chips">
                  {model.top_numbers.slice(0, 5).map(({ number: n, probability: p }) => (
                    <div key={n} className={`number-chip color-${getColor(n)} ${p > 0.1 ? 'hot' : ''}`}>
                      <span className="chip-number">{n}</span>
                      <span className="chip-prob">{(p * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* AI Betting Areas Predictions */}
      {data.betting_areas && (
        <div className="betting-section ai-betting">
          <h3>🤖 Predizioni AI - Aree di Scommessa</h3>
          <p className="section-desc">Previsioni basate sull'ensemble di modelli ML</p>
          <div className="betting-predictions-grid">
            {/* Dozen */}
            <div className="betting-prediction-card">
              <h4>Dozzine</h4>
              <div className="prediction-badge">
                🎯 {data.betting_areas.dozen.prediction}
              </div>
              <div className="confidence-mini">
                Confidenza: {(data.betting_areas.dozen.confidence * 100).toFixed(0)}%
              </div>
              <div className="betting-probs">
                {Object.entries(data.betting_areas.dozen.probabilities).map(([name, prob]) => (
                  <div key={name} className="betting-prob-row">
                    <span className="betting-prob-label">{name}</span>
                    <div className="betting-prob-bar-container">
                      <div className="betting-prob-bar" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Column */}
            <div className="betting-prediction-card">
              <h4>Colonne</h4>
              <div className="prediction-badge">
                🎯 {data.betting_areas.column.prediction}
              </div>
              <div className="confidence-mini">
                Confidenza: {(data.betting_areas.column.confidence * 100).toFixed(0)}%
              </div>
              <div className="betting-probs">
                {Object.entries(data.betting_areas.column.probabilities).map(([name, prob]) => (
                  <div key={name} className="betting-prob-row">
                    <span className="betting-prob-label">{name}</span>
                    <div className="betting-prob-bar-container">
                      <div className="betting-prob-bar" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* High/Low */}
            <div className="betting-prediction-card">
              <h4>Alto / Basso</h4>
              <div className="prediction-badge">
                🎯 {data.betting_areas.high_low.prediction}
              </div>
              <div className="confidence-mini">
                Confidenza: {(data.betting_areas.high_low.confidence * 100).toFixed(0)}%
              </div>
              <div className="betting-probs">
                {Object.entries(data.betting_areas.high_low.probabilities).map(([name, prob]) => (
                  <div key={name} className="betting-prob-row">
                    <span className="betting-prob-label">{name}</span>
                    <div className="betting-prob-bar-container">
                      <div className="betting-prob-bar" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Parity */}
            <div className="betting-prediction-card">
              <h4>Pari / Dispari</h4>
              <div className="prediction-badge">
                🎯 {data.betting_areas.parity.prediction}
              </div>
              <div className="confidence-mini">
                Confidenza: {(data.betting_areas.parity.confidence * 100).toFixed(0)}%
              </div>
              <div className="betting-probs">
                {Object.entries(data.betting_areas.parity.probabilities).map(([name, prob]) => (
                  <div key={name} className="betting-prob-row">
                    <span className="betting-prob-label">{name}</span>
                    <div className="betting-prob-bar-container">
                      <div className="betting-prob-bar" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {data.betting_areas.zero_probability > 0.05 && (
            <div className="zero-warning">
              ⚠️ Attenzione: Probabilità Zero alta ({(data.betting_areas.zero_probability * 100).toFixed(1)}%)
            </div>
          )}
        </div>
      )}

      {/* Hint messages */}
      {total_spins > 0 && total_spins < 30 && !data.betting_areas && (
        <p className="tip">
          💡 Aggiungi altre {30 - total_spins} uscite per abilitare le previsioni AI sulle aree di scommessa
        </p>
      )}
    </div>
  );
}

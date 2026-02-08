import { AdvancedPredictionsResponse } from '../api';

interface Props {
    data: AdvancedPredictionsResponse | null;
    loading: boolean;
    error: string | null;
    onRetry?: () => void;
}

const colorMap: Record<string, string> = {
    red: '#ef4444',
    black: '#4b5563', // Grey-600 for better visibility on dark bg
    green: '#22c55e',
};

export function AdvancedPredictions({ data, loading, error, onRetry }: Props) {
    if (loading) {
        return (
            <div className="card loading-state">
                <div className="spinner"></div>
                <p>Elaborazione modelli AI avanzati...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="card error-state">
                <span className="error-icon">⚠️</span>
                <p>{error}</p>
                {onRetry && (
                    <button type="button" className="retry-button" onClick={onRetry}>
                        Riprova
                    </button>
                )}
            </div>
        );
    }

    if (!data || data.error) {
        return (
            <div className="card info-state">
                <span className="info-icon">🤖</span>
                <p>{data?.error || 'Servono almeno 30 uscite per le predizioni AI avanzate'}</p>
                <p className="subtext">
                    {(data?.total_spins || 0) === 0
                        ? 'Inserisci i numeri che escono alla roulette nella casella sopra'
                        : `Uscite registrate: ${data?.total_spins || 0} / 30`
                    }
                </p>
            </div>
        );
    }

    return (
        <div className="advanced-predictions">
            <h2 className="section-title">
                <span className="icon">🧠</span>
                Predizioni AI Ensemble
            </h2>

            {/* Color Predictions */}
            {data.color && (() => {
                const sortedColors = Object.entries(data.color.ensemble).sort(([, a], [, b]) => b - a);
                const predictedColor = sortedColors[0][0];

                return (
                    <div className="prediction-section">
                        <div className="section-header-row">
                            <h3>Predizione Colore</h3>
                        </div>

                        <div className="prediction-main" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '1.5rem' }}>
                            <span
                                className="prediction-badge-large"
                                style={{
                                    backgroundColor: colorMap[predictedColor],
                                    boxShadow: `0 0 25px ${predictedColor === 'black' ? 'rgba(255, 255, 255, 0.3)' : colorMap[predictedColor]}`,
                                    color: 'white',
                                    marginBottom: '1rem',
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    padding: '0.8rem 2.5rem',
                                    borderRadius: '9999px',
                                    fontSize: '1.5rem',
                                    fontWeight: '800',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    border: '2px solid rgba(255, 255, 255, 0.2)'
                                }}
                            >
                                {predictedColor.toUpperCase()}
                            </span>

                            <div className="agreement-badge" style={{
                                width: 'fit-content',
                                padding: '0.3rem 1rem',
                                borderRadius: '9999px',
                                background: 'rgba(255, 255, 255, 0.05)',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                fontSize: '0.85rem'
                            }}>
                                <span>Accordo AI: </span>
                                <strong style={{ color: 'white', marginLeft: '4px' }}>{(data.color.agreement * 100).toFixed(0)}%</strong>
                            </div>
                        </div>

                        <div className="capsule-list">
                            {['red', 'black', 'green'].map(color => {
                                const prob = data.color!.ensemble[color] || 0;
                                const isBlack = color === 'black';
                                const colorCode = colorMap[color];
                                const glowColor = isBlack ? 'rgba(255, 255, 255, 0.4)' : colorCode;

                                return (
                                    <div key={color} className="prob-row-capsule">
                                        <div
                                            className="capsule-dot"
                                            style={{
                                                backgroundColor: colorCode,
                                                color: glowColor
                                            }}
                                        />
                                        <div className="capsule-label">{color}</div>
                                        <div className="capsule-bar-track">
                                            <div
                                                className="capsule-bar-fill"
                                                style={{
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: colorCode,
                                                    color: glowColor
                                                }}
                                            />
                                        </div>
                                        <div className="capsule-value">{(prob * 100).toFixed(1)}%</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );
            })()}

            {/* Number Predictions */}
            {data.number && data.number.ensemble.length > 0 && (() => {
                const topNumber = data.number.ensemble[0];
                const topColor = topNumber.number === 0 ? 'green'
                    : [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36].includes(topNumber.number)
                        ? 'red' : 'black';
                return (
                    <div className="prediction-section">
                        <h3>Predizione Numeri</h3>

                        <div className="betting-probability-box betting-probability-box-number">
                            <div
                                className="number-circle number-circle-large"
                                style={{ backgroundColor: colorMap[topColor] }}
                            >
                                {topNumber.number}
                            </div>
                            <div className="betting-probability-content">
                                <span className="betting-probability-label">Probabilità stimata (per scommessa sul numero)</span>
                                <span className="betting-probability-value">{(topNumber.probability * 100).toFixed(2)}%</span>
                            </div>
                            <p className="betting-probability-hint">
                                Probabilità che l’ensemble assegna al numero più probabile. Sotto la top 10 completa.
                            </p>
                        </div>

                        <div className="ensemble-result">
                            <div className="agreement-badge">
                                <span>Accordo modelli: </span>
                                <strong>{(data.number.agreement * 100).toFixed(0)}%</strong>
                                <span className="agreement-hint"> — su top numero</span>
                            </div>
                        </div>

                        <div className="top-numbers">
                            <div className="confidence-header">
                                <span className="label">Top 10 numeri per probabilità stimata</span>
                            </div>

                            <div className="numbers-grid">
                                {data.number.ensemble.slice(0, 10).map((item, idx) => {
                                    const color = item.number === 0 ? 'green'
                                        : [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36].includes(item.number)
                                            ? 'red' : 'black';

                                    return (
                                        <div
                                            key={item.number}
                                            className={`number-prediction ${idx === 0 ? 'top-pick' : ''}`}
                                        >
                                            <div
                                                className="number-circle"
                                                style={{ backgroundColor: colorMap[color] }}
                                            >
                                                {item.number}
                                            </div>
                                            <div className="number-prob">
                                                {(item.probability * 100).toFixed(2)}%
                                            </div>
                                            {idx === 0 && <span className="top-badge">🎯 TOP</span>}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                );
            })()}

            {/* Betting Areas Predictions */}
            {data.betting_areas && (
                <div className="prediction-section">
                    <h3>🎯 Predizioni Aree di Scommessa</h3>
                    <p className="section-desc">Previsioni AI basate sulla distribuzione di probabilità dei numeri</p>

                    <div className="betting-predictions-grid">
                        {/* Dozens */}
                        <div className="betting-prediction-card">
                            <h4>Dozzine</h4>
                            <div className="prediction-badge">
                                🎯 {data.betting_areas.dozen.prediction}
                            </div>
                            <div className="ensemble-result" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                                <div className="agreement-badge">
                                    <span>Accordo: </span>
                                    <strong>{(data.betting_areas.dozen.agreement * 100).toFixed(0)}%</strong>
                                </div>
                            </div>
                            <div className="confidence-mini" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
                                Confidenza: {(data.betting_areas.dozen.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.dozen.probabilities).map(([name, prob]) => (
                                    <div key={name} className="prob-row-capsule" style={{ gridTemplateColumns: '24px 60px 1fr 40px', padding: '0.5rem 0.8rem' }}>
                                        <div className="capsule-dot" style={{ backgroundColor: 'var(--accent-primary)', color: 'var(--accent-glow)' }} />
                                        <div className="capsule-label" style={{ fontSize: '0.7rem' }}>{name}</div>
                                        <div className="capsule-bar-track">
                                            <div
                                                className="capsule-bar-fill"
                                                style={{
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: 'var(--accent-primary)',
                                                    color: 'var(--accent-glow)'
                                                }}
                                            />
                                        </div>
                                        <div className="capsule-value" style={{ fontSize: '0.8rem' }}>{(prob * 100).toFixed(0)}%</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Columns */}
                        <div className="betting-prediction-card">
                            <h4>Colonne</h4>
                            <div className="prediction-badge">
                                🎯 {data.betting_areas.column.prediction}
                            </div>
                            <div className="ensemble-result" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                                <div className="agreement-badge">
                                    <span>Accordo: </span>
                                    <strong>{(data.betting_areas.column.agreement * 100).toFixed(0)}%</strong>
                                </div>
                            </div>
                            <div className="confidence-mini" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
                                Confidenza: {(data.betting_areas.column.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.column.probabilities).map(([name, prob]) => (
                                    <div key={name} className="prob-row-capsule" style={{ gridTemplateColumns: '24px 60px 1fr 40px', padding: '0.5rem 0.8rem' }}>
                                        <div className="capsule-dot" style={{ backgroundColor: 'var(--accent-secondary)', color: 'rgba(167, 139, 250, 0.4)' }} />
                                        <div className="capsule-label" style={{ fontSize: '0.7rem' }}>{name}</div>
                                        <div className="capsule-bar-track">
                                            <div
                                                className="capsule-bar-fill"
                                                style={{
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: 'var(--accent-secondary)',
                                                    color: 'rgba(167, 139, 250, 0.4)'
                                                }}
                                            />
                                        </div>
                                        <div className="capsule-value" style={{ fontSize: '0.8rem' }}>{(prob * 100).toFixed(0)}%</div>
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
                            <div className="ensemble-result" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                                <div className="agreement-badge">
                                    <span>Accordo: </span>
                                    <strong>{(data.betting_areas.high_low.agreement * 100).toFixed(0)}%</strong>
                                </div>
                            </div>
                            <div className="confidence-mini" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
                                Confidenza: {(data.betting_areas.high_low.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.high_low.probabilities).map(([name, prob]) => (
                                    <div key={name} className="prob-row-capsule" style={{ gridTemplateColumns: '24px 60px 1fr 40px', padding: '0.5rem 0.8rem' }}>
                                        <div className="capsule-dot" style={{ backgroundColor: '#f472b6', color: 'rgba(244, 114, 182, 0.4)' }} />
                                        <div className="capsule-label" style={{ fontSize: '0.7rem' }}>{name}</div>
                                        <div className="capsule-bar-track">
                                            <div
                                                className="capsule-bar-fill"
                                                style={{
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: '#f472b6',
                                                    color: 'rgba(244, 114, 182, 0.4)'
                                                }}
                                            />
                                        </div>
                                        <div className="capsule-value" style={{ fontSize: '0.8rem' }}>{(prob * 100).toFixed(0)}%</div>
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
                            <div className="ensemble-result" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                                <div className="agreement-badge">
                                    <span>Accordo: </span>
                                    <strong>{(data.betting_areas.parity.agreement * 100).toFixed(0)}%</strong>
                                </div>
                            </div>
                            <div className="confidence-mini" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
                                Confidenza: {(data.betting_areas.parity.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.parity.probabilities).map(([name, prob]) => (
                                    <div key={name} className="prob-row-capsule" style={{ gridTemplateColumns: '24px 60px 1fr 40px', padding: '0.5rem 0.8rem' }}>
                                        <div className="capsule-dot" style={{ backgroundColor: '#818cf8', color: 'rgba(129, 140, 248, 0.4)' }} />
                                        <div className="capsule-label" style={{ fontSize: '0.7rem' }}>{name}</div>
                                        <div className="capsule-bar-track">
                                            <div
                                                className="capsule-bar-fill"
                                                style={{
                                                    width: `${prob * 100}%`,
                                                    backgroundColor: '#818cf8',
                                                    color: 'rgba(129, 140, 248, 0.4)'
                                                }}
                                            />
                                        </div>
                                        <div className="capsule-value" style={{ fontSize: '0.8rem' }}>{(prob * 100).toFixed(0)}%</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {data.betting_areas.zero_probability > 0.05 && (
                        <div className="zero-warning">
                            ⚠️ Probabilità Zero: {(data.betting_areas.zero_probability * 100).toFixed(1)}% - considera lo zero!
                        </div>
                    )}
                </div>
            )}

        </div>
    );
}

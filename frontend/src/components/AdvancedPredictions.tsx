import { AdvancedPredictionsResponse } from '../api';

interface Props {
    data: AdvancedPredictionsResponse | null;
    loading: boolean;
    error: string | null;
}

const colorMap: Record<string, string> = {
    red: '#ef4444',
    black: '#1f2937',
    green: '#22c55e',
};

const modelColors: Record<string, string> = {
    DeepMLP: '#8b5cf6',
    RandomForest: '#10b981',
    GradientBoosting: '#f59e0b',
    XGBoost: '#ec4899',
};

export function AdvancedPredictions({ data, loading, error }: Props) {
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
            {data.color && (
                <div className="prediction-section">
                    <h3>Predizione Colore</h3>

                    {/* Ensemble Result */}
                    <div className="ensemble-result">
                        <div className="confidence-header">
                            <span className="label">Confidenza Ensemble</span>
                            <span className="value">{(data.color.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="confidence-bar">
                            <div
                                className="confidence-fill"
                                style={{ width: `${data.color.confidence * 100}%` }}
                            />
                        </div>

                        <div className="agreement-badge">
                            <span>Accordo modelli: </span>
                            <strong>{(data.color.agreement * 100).toFixed(0)}%</strong>
                        </div>
                    </div>

                    {/* Color Probabilities */}
                    <div className="color-probs">
                        {Object.entries(data.color.ensemble)
                            .sort(([, a], [, b]) => b - a)
                            .map(([color, prob]) => (
                                <div key={color} className="color-prob-item">
                                    <div
                                        className="color-dot"
                                        style={{ backgroundColor: colorMap[color] }}
                                    />
                                    <span className="color-name">{color}</span>
                                    <div className="prob-bar-container">
                                        <div
                                            className="prob-bar"
                                            style={{
                                                width: `${prob * 100}%`,
                                                backgroundColor: colorMap[color],
                                            }}
                                        />
                                    </div>
                                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                    </div>

                    {/* Individual Models */}
                    <div className="models-comparison">
                        <h4>Confronto Modelli</h4>
                        <div className="models-grid">
                            {Object.entries(data.color.models).map(([modelName, probs]) => {
                                const topColor = Object.entries(probs).sort(([, a], [, b]) => b - a)[0];
                                const weight = data.color?.weights?.[modelName] || 0;

                                return (
                                    <div key={modelName} className="model-card">
                                        <div className="model-header">
                                            <span
                                                className="model-indicator"
                                                style={{ backgroundColor: modelColors[modelName] || '#6b7280' }}
                                            />
                                            <span className="model-name">{modelName}</span>
                                        </div>
                                        <div className="model-prediction">
                                            <div
                                                className="predicted-color"
                                                style={{ backgroundColor: colorMap[topColor[0]] }}
                                            />
                                            <span>{(topColor[1] * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="model-weight">
                                            Peso: {(weight * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* Number Predictions */}
            {data.number && (
                <div className="prediction-section">
                    <h3>Predizione Numeri</h3>

                    <div className="top-numbers">
                        <div className="confidence-header">
                            <span className="label">Top Numeri Predetti</span>
                            <span className="value">Confidenza: {(data.number.confidence * 100).toFixed(1)}%</span>
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
            )}

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
                            <div className="confidence-mini">
                                Conf: {(data.betting_areas.dozen.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.dozen.probabilities).map(([name, prob]) => (
                                    <div key={name} className="betting-prob-row">
                                        <span className="betting-prob-label">{name}</span>
                                        <div className="betting-prob-bar-container">
                                            <div
                                                className="betting-prob-bar"
                                                style={{ width: `${prob * 100}%` }}
                                            />
                                        </div>
                                        <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
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
                            <div className="confidence-mini">
                                Conf: {(data.betting_areas.column.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.column.probabilities).map(([name, prob]) => (
                                    <div key={name} className="betting-prob-row">
                                        <span className="betting-prob-label">{name}</span>
                                        <div className="betting-prob-bar-container">
                                            <div
                                                className="betting-prob-bar"
                                                style={{ width: `${prob * 100}%` }}
                                            />
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
                                Conf: {(data.betting_areas.high_low.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.high_low.probabilities).map(([name, prob]) => (
                                    <div key={name} className="betting-prob-row">
                                        <span className="betting-prob-label">{name}</span>
                                        <div className="betting-prob-bar-container">
                                            <div
                                                className="betting-prob-bar"
                                                style={{ width: `${prob * 100}%` }}
                                            />
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
                                Conf: {(data.betting_areas.parity.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="betting-probs">
                                {Object.entries(data.betting_areas.parity.probabilities).map(([name, prob]) => (
                                    <div key={name} className="betting-prob-row">
                                        <span className="betting-prob-label">{name}</span>
                                        <div className="betting-prob-bar-container">
                                            <div
                                                className="betting-prob-bar"
                                                style={{ width: `${prob * 100}%` }}
                                            />
                                        </div>
                                        <span className="betting-prob-value">{(prob * 100).toFixed(0)}%</span>
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

            {/* Model Info */}
            {data.model_info && (
                <div className="model-info-section">
                    <h4>Stato Modelli</h4>
                    <div className="model-status-grid">
                        {Object.entries(data.model_info.models).map(([name, info]) => (
                            <div key={name} className={`model-status ${info.trained ? 'trained' : 'untrained'}`}>
                                <span className="status-dot" />
                                <span className="model-name">{name}</span>
                                <span className="status-text">
                                    {info.trained ? 'Attivo' : info.available ? 'Non addestrato' : 'Non disponibile'}
                                </span>
                            </div>
                        ))}
                    </div>
                    <p className="samples-info">
                        Campioni totali: {data.model_info.total_samples} |
                        Riaddestr. ogni {data.model_info.retrain_interval} spin
                    </p>
                </div>
            )}
        </div>
    );
}

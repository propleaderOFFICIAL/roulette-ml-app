import { StatisticalAnalysisResponse } from '../api';

interface Props {
    data: StatisticalAnalysisResponse | null;
    loading: boolean;
    error: string | null;
}

export function StatisticsPanel({ data, loading, error }: Props) {
    if (loading) {
        return (
            <div className="card loading-state">
                <div className="spinner"></div>
                <p>Calcolo statistiche avanzate...</p>
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
                <span className="info-icon">📐</span>
                <p>{data?.error || 'Servono almeno 20 spin per analisi statistica'}</p>
            </div>
        );
    }

    const { statistics: stats } = data;

    return (
        <div className="statistics-panel">
            <h2 className="section-title">
                <span className="icon">📊</span>
                Analisi Statistica Avanzata
            </h2>

            {/* Overall Assessment */}
            <div className="overall-assessment">
                <div className={`assessment-badge bias-${stats.bias_indicators}`}>
                    <span className="assessment-icon">
                        {stats.bias_indicators >= 3 ? '🔴' : stats.bias_indicators >= 2 ? '🟡' : '🟢'}
                    </span>
                    <span className="assessment-text">{stats.overall_assessment}</span>
                </div>
                <span className="bias-count">Indicatori bias: {stats.bias_indicators}/4</span>
            </div>

            {/* Chi-Squared Tests */}
            <div className="stats-section chi-squared">
                <h3>🧪 Test Chi-Quadrato</h3>
                <p className="section-desc">Verifica se la distribuzione è uniforme come atteso</p>

                <div className="chi-squared-grid">
                    <div className={`chi-squared-card ${stats.chi_squared.number.significant ? 'significant' : ''}`}>
                        <h4>Distribuzione Numeri</h4>
                        <div className="stat-value">χ² = {stats.chi_squared.number.statistic.toFixed(2)}</div>
                        <div className="p-value">p-value: {stats.chi_squared.number.p_value.toFixed(4)}</div>
                        <div className="interpretation">{stats.chi_squared.number.interpretation}</div>
                    </div>

                    <div className={`chi-squared-card ${stats.chi_squared.color.significant ? 'significant' : ''}`}>
                        <h4>Distribuzione Colori</h4>
                        <div className="stat-value">χ² = {stats.chi_squared.color.statistic.toFixed(2)}</div>
                        <div className="p-value">p-value: {stats.chi_squared.color.p_value.toFixed(4)}</div>
                        <div className="interpretation">{stats.chi_squared.color.interpretation}</div>
                    </div>
                </div>
            </div>

            {/* Entropy */}
            <div className="stats-section entropy">
                <h3>🎲 Entropia e Casualità</h3>

                <div className="entropy-meter">
                    <div className="meter-label">
                        <span>Bassa</span>
                        <span>Casualità</span>
                        <span>Alta</span>
                    </div>
                    <div className="meter-bar">
                        <div
                            className="meter-fill"
                            style={{
                                width: `${stats.entropy.randomness_score * 100}%`,
                                backgroundColor: getRandomnessColor(stats.entropy.randomness_score),
                            }}
                        />
                        <div
                            className="meter-marker"
                            style={{ left: `${stats.entropy.randomness_score * 100}%` }}
                        >
                            {(stats.entropy.randomness_score * 100).toFixed(0)}%
                        </div>
                    </div>
                </div>

                <div className="entropy-details">
                    <span>Entropia numeri: {stats.entropy.number_entropy.toFixed(3)}</span>
                    <span>Entropia colori: {stats.entropy.color_entropy.toFixed(3)}</span>
                </div>
                <p className="interpretation">{stats.entropy.interpretation}</p>
            </div>

            {/* Bayesian */}
            <div className="stats-section bayesian">
                <h3>📈 Inferenza Bayesiana</h3>
                <p className="section-desc">Probabilità aggiornate basate sui dati osservati</p>

                <div className="bayesian-grid">
                    <div className="bayesian-comparison">
                        <h4>Distribuzione A Priori vs Posteriore</h4>
                        {['red', 'black', 'green'].map((color) => (
                            <div key={color} className="distribution-row">
                                <span className="color-label" style={{ color: getColorHex(color) }}>
                                    {color}
                                </span>
                                <div className="bar-comparison">
                                    <div className="bar-container">
                                        <div
                                            className="bar prior"
                                            style={{ width: `${stats.bayesian.prior[color] * 100}%` }}
                                        />
                                        <span className="bar-label">Prior: {(stats.bayesian.prior[color] * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="bar-container">
                                        <div
                                            className="bar posterior"
                                            style={{ width: `${stats.bayesian.posterior[color] * 100}%` }}
                                        />
                                        <span className="bar-label">Post: {(stats.bayesian.posterior[color] * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="bayesian-prediction">
                        <span className="prediction-label">Predizione Bayesiana:</span>
                        <span
                            className="predicted-color"
                            style={{ backgroundColor: getColorHex(stats.bayesian.predicted_color) }}
                        >
                            {stats.bayesian.predicted_color}
                        </span>
                        <span className="confidence">
                            Confidenza: {(stats.bayesian.confidence * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* Markov Chain */}
            <div className="stats-section markov">
                <h3>🔗 Analisi Catena di Markov</h3>
                <p className="section-desc">Probabilità di transizione tra stati</p>

                <div className="markov-current">
                    <span>Stato attuale: </span>
                    <span className="current-color" style={{ color: getColorHex(stats.markov.current_state.last_color) }}>
                        {stats.markov.current_state.last_color}
                    </span>
                </div>

                <div className="transition-matrix">
                    <h4>Predizione prossimo colore (da stato attuale):</h4>
                    <div className="next-prediction">
                        {Object.entries(stats.markov.next_color_prediction)
                            .sort(([, a], [, b]) => b - a)
                            .map(([color, prob]) => (
                                <div key={color} className="transition-item">
                                    <span
                                        className="transition-color"
                                        style={{ backgroundColor: getColorHex(color) }}
                                    />
                                    <span className="transition-label">{color}</span>
                                    <div className="transition-bar">
                                        <div
                                            className="transition-fill"
                                            style={{ width: `${prob * 100}%`, backgroundColor: getColorHex(color) }}
                                        />
                                    </div>
                                    <span className="transition-prob">{(prob * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                    </div>
                </div>
            </div>

            {/* Monte Carlo */}
            <div className="stats-section monte-carlo">
                <h3>🎰 Simulazione Monte Carlo</h3>
                <p className="section-desc">{stats.monte_carlo.color.iterations?.toLocaleString()} iterazioni simulate</p>

                <div className="monte-carlo-comparison">
                    <div className="mc-column">
                        <h4>Teorico vs Simulato (Colori)</h4>
                        {stats.monte_carlo.color.simulated_probabilities && (
                            <div className="mc-bars">
                                {['red', 'black', 'green'].map((color) => (
                                    <div key={color} className="mc-row">
                                        <span className="mc-label" style={{ color: getColorHex(color) }}>{color}</span>
                                        <div className="mc-bar-group">
                                            <div className="mc-bar-container">
                                                <div
                                                    className="mc-bar theoretical"
                                                    style={{
                                                        width: `${(stats.monte_carlo.color.theoretical_probabilities?.[color] || 0) * 100}%`
                                                    }}
                                                />
                                            </div>
                                            <div className="mc-bar-container">
                                                <div
                                                    className="mc-bar simulated"
                                                    style={{
                                                        width: `${(stats.monte_carlo.color.simulated_probabilities?.[color] || 0) * 100}%`
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                <div className="mc-legend">
                                    <span className="legend-item"><span className="dot theoretical" /> Teorico</span>
                                    <span className="legend-item"><span className="dot simulated" /> Simulato</span>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Runs Test */}
            <div className="stats-section runs-test">
                <h3>🏃 Test delle Serie (Wald-Wolfowitz)</h3>

                <div className="runs-result">
                    <div className="runs-stat">
                        <span className="stat-label">Serie osservate:</span>
                        <span className="stat-value">{stats.runs_test.runs_observed}</span>
                    </div>
                    <div className="runs-stat">
                        <span className="stat-label">Serie attese:</span>
                        <span className="stat-value">{stats.runs_test.runs_expected.toFixed(1)}</span>
                    </div>
                    <div className="runs-stat">
                        <span className="stat-label">Z-score:</span>
                        <span className="stat-value">{stats.runs_test.z_score.toFixed(3)}</span>
                    </div>
                    <div className="runs-stat">
                        <span className="stat-label">p-value:</span>
                        <span className="stat-value">{stats.runs_test.p_value.toFixed(4)}</span>
                    </div>
                </div>

                <div className={`runs-conclusion ${stats.runs_test.random ? 'random' : 'not-random'}`}>
                    {stats.runs_test.random ? '✅' : '⚠️'} {stats.runs_test.interpretation}
                </div>
            </div>
        </div>
    );
}

function getRandomnessColor(score: number): string {
    if (score > 0.85) return '#22c55e';
    if (score > 0.70) return '#84cc16';
    if (score > 0.50) return '#f59e0b';
    return '#ef4444';
}

function getColorHex(color: string): string {
    const colors: Record<string, string> = {
        red: '#ef4444',
        black: '#1f2937',
        green: '#22c55e',
    };
    return colors[color] || '#6b7280';
}

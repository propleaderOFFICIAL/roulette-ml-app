import { StatisticalAnalysisResponse } from '../api';

interface Props {
    data: StatisticalAnalysisResponse | null;
    loading: boolean;
    error: string | null;
}

export function StatisticsPanel({ data, loading, error }: Props) {
    if (loading) {
        return (
            <div className="stat-card-glass loading-state" style={{ minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div className="spinner"></div>
                <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Calcolo statistiche avanzate...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="stat-card-glass error-state">
                <span className="error-icon" style={{ fontSize: '2rem' }}>⚠️</span>
                <p>{error}</p>
            </div>
        );
    }

    if (!data || data.error) {
        return (
            <div className="stat-card-glass info-state">
                <span className="info-icon" style={{ fontSize: '2rem' }}>📐</span>
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

            <div className="info-box-premium">
                <h4 style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    💡 A cosa servono queste statistiche
                </h4>
                <p style={{ margin: 0, lineHeight: 1.6 }}>
                    Qui non si “prevede” il prossimo numero, ma si analizza se le uscite che hai registrato
                    assomigliano a una roulette equilibrata (casuale) o se emergono pattern strani.
                    I valori sotto ti dicono <strong>come leggere</strong> ogni blocco in modo semplice.
                </p>
            </div>

            {/* Overall Assessment */}
            <div className="stat-card-glass overall-assessment">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h3 className="stat-section-title" style={{ margin: 0, fontSize: '1.2rem' }}>Valutazione Globale</h3>
                    <span className="bias-count" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                        Bias rilevati: <strong style={{ color: 'white' }}>{stats.bias_indicators}/4</strong>
                    </span>
                </div>

                <div className={`assessment-badge bias-${stats.bias_indicators}`} style={{
                    padding: '1.5rem',
                    borderRadius: 'var(--radius-lg)',
                    background: 'rgba(0,0,0,0.3)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem'
                }}>
                    <span className="assessment-icon" style={{ fontSize: '2.5rem' }}>
                        {stats.bias_indicators >= 3 ? '🔴' : stats.bias_indicators >= 2 ? '🟡' : '🟢'}
                    </span>
                    <div>
                        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1.2rem', color: 'white' }}>{stats.overall_assessment}</h4>
                        <p style={{ margin: 0, fontSize: '0.9rem', opacity: 0.8 }}>
                            Il semaforo indica la salute della sessione. Verde = tutto nella norma.
                        </p>
                    </div>
                </div>
            </div>

            <div className="stats-grid-2">
                {/* Chi-Squared Tests */}
                <div className="stat-card-glass chi-squared">
                    <h3 className="stat-section-title"><span className="icon">🧪</span> Test Chi-Quadrato</h3>
                    <p className="stat-label-small" style={{ marginBottom: '1rem' }}>Uniformità Distribuzione</p>

                    <div style={{ display: 'grid', gap: '1rem' }}>
                        {/* Number Stats */}
                        <div className={`chi-squared-card ${stats.chi_squared.number.significant ? 'significant' : ''}`} style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Numeri</span>
                                <span className={stats.chi_squared.number.significant ? 'status-pill danger' : 'status-pill success'} style={{ fontSize: '0.7rem', padding: '0.2rem 0.6rem' }}>
                                    {stats.chi_squared.number.significant ? 'Sospetto' : 'Ok'}
                                </span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                                <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'white' }}>{stats.chi_squared.number.p_value.toFixed(4)}</span>
                                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>p-value</span>
                            </div>
                            <div style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.8 }}>{stats.chi_squared.number.interpretation}</div>
                        </div>

                        {/* Color Stats */}
                        <div className={`chi-squared-card ${stats.chi_squared.color.significant ? 'significant' : ''}`} style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Colori</span>
                                <span className={stats.chi_squared.color.significant ? 'status-pill danger' : 'status-pill success'} style={{ fontSize: '0.7rem', padding: '0.2rem 0.6rem' }}>
                                    {stats.chi_squared.color.significant ? 'Sospetto' : 'Ok'}
                                </span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                                <span style={{ fontSize: '1.5rem', fontWeight: 800, color: 'white' }}>{stats.chi_squared.color.p_value.toFixed(4)}</span>
                                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>p-value</span>
                            </div>
                            <div style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.8 }}>{stats.chi_squared.color.interpretation}</div>
                        </div>
                    </div>
                </div>

                {/* Entropy */}
                <div className="stat-card-glass entropy">
                    <h3 className="stat-section-title"><span className="icon">🎲</span> Entropia</h3>
                    <p className="stat-label-small">Casualità (Score 0-1)</p>

                    <div className="stat-big-value-display" style={{ margin: '1rem 0' }}>
                        <div className="stat-value-big">{(stats.entropy.randomness_score * 100).toFixed(0)}%</div>
                        <span className="stat-label-small">Randomness Score</span>
                    </div>

                    <div className="premium-meter-container">
                        <div
                            className="premium-meter-fill"
                            style={{
                                width: `${stats.entropy.randomness_score * 100}%`,
                                color: getRandomnessColor(stats.entropy.randomness_score)
                            }}
                        />
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        <span>Bassa</span>
                        <span>Alta</span>
                    </div>

                    <p style={{ marginTop: '1rem', fontSize: '0.9rem', lineHeight: 1.5, opacity: 0.9 }}>
                        {stats.entropy.interpretation}
                    </p>
                </div>
            </div>

            {/* Bayesian */}
            <div className="stat-card-glass bayesian">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                        <h3 className="stat-section-title"><span className="icon">📈</span> Inferenza Bayesiana</h3>
                        <p className="stat-label-small">Probabilità aggiornate post-osservazione</p>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                        <div className="stat-label-small">Predizione</div>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.2rem' }}>
                            <span
                                className="predicted-color-chip"
                                style={{
                                    backgroundColor: getColorHex(stats.bayesian.predicted_color),
                                    boxShadow: `0 0 10px ${getColorHex(stats.bayesian.predicted_color)}`
                                }}
                            >
                                {stats.bayesian.predicted_color}
                            </span>
                            <span style={{ fontWeight: 700, fontSize: '1.2rem', color: 'white' }}>
                                {(stats.bayesian.confidence * 100).toFixed(0)}%
                            </span>
                        </div>
                    </div>
                </div>

                <div style={{ marginTop: '1.5rem' }}>
                    {['red', 'black'].map((color) => (
                        <div key={color} className="prob-row-capsule">
                            <div className="capsule-dot" style={{ backgroundColor: getColorHex(color), color: getColorHex(color) }}></div>
                            <span className="capsule-label" style={{
                                color: color === 'black' ? '#9ca3af' : getColorHex(color),
                                textShadow: color === 'black' ? 'none' : `0 0 10px ${getColorHex(color)}`
                            }}>{color}</span>

                            <div className="capsule-bar-track">
                                <div
                                    className="capsule-bar-fill"
                                    style={{
                                        width: `${stats.bayesian.posterior[color] * 100}%`,
                                        backgroundColor: getColorHex(color),
                                        boxShadow: `0 0 10px ${getColorHex(color)}`,
                                        color: getColorHex(color) // properly inherit color for box-shadow
                                    }}
                                />
                            </div>

                            <div className="capsule-value">
                                {(stats.bayesian.posterior[color] * 100).toFixed(1)}%
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Markov Chain */}
            <div className="stat-card-glass markov">
                <h3 className="stat-section-title"><span className="icon">🔗</span> Catena di Markov</h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                    <span className="stat-label-small">Stato attuale (ultimo):</span>
                    <span
                        className="predicted-color-chip"
                        style={{
                            padding: '0.2rem 0.8rem',
                            fontSize: '0.8rem',
                            backgroundColor: getColorHex(stats.markov.current_state.last_color),
                            boxShadow: `0 0 8px ${getColorHex(stats.markov.current_state.last_color)}`
                        }}
                    >
                        {stats.markov.current_state.last_color}
                    </span>
                    <span className="stat-label-small">→ Probabile prossimo:</span>
                </div>

                <div>
                    {Object.entries(stats.markov.next_color_prediction)
                        .sort(([, a], [, b]) => b - a)
                        .map(([color, prob]) => (
                            <div key={color} className="prob-row-capsule">
                                <div className="capsule-dot" style={{ backgroundColor: getColorHex(color), color: getColorHex(color) }}></div>
                                <span className="capsule-label" style={{
                                    color: color === 'black' ? '#9ca3af' : getColorHex(color)
                                }}>{color}</span>

                                <div className="capsule-bar-track">
                                    <div
                                        className="capsule-bar-fill"
                                        style={{
                                            width: `${prob * 100}%`,
                                            backgroundColor: getColorHex(color),
                                            boxShadow: `0 0 10px ${getColorHex(color)}`,
                                            color: getColorHex(color)
                                        }}
                                    />
                                </div>

                                <span className="capsule-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                </div>
            </div>

            {/* Monte Carlo */}
            <div className="stat-card-glass monte-carlo">
                <h3 className="stat-section-title"><span className="icon">🎰</span> Monte Carlo</h3>
                <p className="stat-label-small" style={{ marginBottom: '1.5rem' }}>
                    {stats.monte_carlo.color.iterations?.toLocaleString()} iterazioni simulate
                </p>

                {stats.monte_carlo.color.simulated_probabilities && (
                    <div style={{ display: 'grid', gap: '1rem' }}>
                        {['red', 'black'].map((color) => (
                            <div key={color} style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: 'var(--radius-lg)' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <span className="capsule-label" style={{ color: getColorHex(color) }}>{color.toUpperCase()}</span>
                                </div>

                                {/* Theoretical */}
                                <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 50px', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TEORICO</span>
                                    <div className="capsule-bar-track" style={{ height: '4px', background: 'rgba(255,255,255,0.05)' }}>
                                        <div
                                            className="capsule-bar-fill"
                                            style={{
                                                width: `${(stats.monte_carlo.color.theoretical_probabilities?.[color] || 0) * 100}%`,
                                                backgroundColor: 'rgba(255,255,255,0.3)',
                                                boxShadow: 'none'
                                            }}
                                        />
                                    </div>
                                    <span className="capsule-value" style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                                        {((stats.monte_carlo.color.theoretical_probabilities?.[color] || 0) * 100).toFixed(1)}%
                                    </span>
                                </div>

                                {/* Simulated */}
                                <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 50px', alignItems: 'center', gap: '1rem' }}>
                                    <span style={{ fontSize: '0.7rem', color: 'white', fontWeight: 'bold' }}>SIMULATO</span>
                                    <div className="capsule-bar-track">
                                        <div
                                            className="capsule-bar-fill"
                                            style={{
                                                width: `${(stats.monte_carlo.color.simulated_probabilities?.[color] || 0) * 100}%`,
                                                backgroundColor: getColorHex(color),
                                                boxShadow: `0 0 10px ${getColorHex(color)}`,
                                                color: getColorHex(color)
                                            }}
                                        />
                                    </div>
                                    <span className="capsule-value" style={{ color: 'white' }}>
                                        {((stats.monte_carlo.color.simulated_probabilities?.[color] || 0) * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Runs Test */}
            <div className="stat-card-glass runs-test">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <h3 className="stat-section-title" style={{ marginBottom: 0 }}><span className="icon">🏃</span> Test delle Serie</h3>
                    <span className={stats.runs_test.random ? 'status-pill success' : 'status-pill danger'}>
                        {stats.runs_test.random ? '✅ Random' : '⚠️ Non Random'}
                    </span>
                </div>
                <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                        <div className="stat-value-big" style={{ fontSize: '1.8rem' }}>{stats.runs_test.runs_observed}</div>
                        <div className="stat-label-small">Osservate</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div className="stat-value-big" style={{ fontSize: '1.8rem', color: 'var(--text-muted)', background: 'none', WebkitTextFillColor: 'var(--text-muted)' }}>
                            {stats.runs_test.runs_expected.toFixed(1)}
                        </div>
                        <div className="stat-label-small">Attese</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div className="stat-value-big" style={{ fontSize: '1.8rem' }}>{stats.runs_test.z_score.toFixed(2)}</div>
                        <div className="stat-label-small">Z-Score</div>
                    </div>
                </div>
                <p style={{ textAlign: 'center', marginTop: '1rem', opacity: 0.8, fontSize: '0.9rem' }}>
                    {stats.runs_test.interpretation}
                </p>
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
        black: '#4b5563', // Lighter grey for visibility
        green: '#22c55e',
    };
    return colors[color] || '#6b7280';
}

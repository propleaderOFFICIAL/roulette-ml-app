import { WheelClusteringResponse } from '../api';
import './WheelAnalysis.css';

interface Props {
    data: WheelClusteringResponse | null;
    loading: boolean;
    error: string | null;
}

const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

export function WheelAnalysis({ data, loading, error }: Props) {
    if (loading) {
        return (
            <div className="card loading-state">
                <div className="spinner"></div>
                <p>Analisi della ruota in corso...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="card error-state">
                <p>⚠️ {error}</p>
            </div>
        );
    }

    if (!data || data.error) {
        return (
            <div className="card info-state">
                <p>🎰 {data?.error || 'Servono almeno 30 spin per l\'analisi della ruota...'}</p>
            </div>
        );
    }

    const { wheel_analysis } = data;
    const { sector_clustering, pair_analysis, sleeper_anomalies } = wheel_analysis;

    const getColorForNumber = (num: number) => {
        if (num === 0) return '#22c55e';
        return RED_NUMBERS.includes(num) ? '#ef4444' : '#1f2937';
    };

    const getBiasIcon = () => {
        if (wheel_analysis.exploitable) return '🎯';
        if (wheel_analysis.bias_indicators >= 2) return '🔍';
        return '✅';
    };

    const getAssessmentColor = () => {
        if (wheel_analysis.exploitable) return 'var(--color-success)';
        if (wheel_analysis.bias_indicators >= 2) return 'var(--color-warning)';
        return 'var(--color-muted)';
    };

    return (
        <div className="wheel-analysis-container">
            <h2 className="section-title">
                <span className="icon">🎰</span> Analisi Fisica della Ruota
            </h2>

            {/* MAIN ASSESSMENT CARD */}
            <div className={`assessment-card ${wheel_analysis.exploitable ? 'exploitable' : ''}`}>
                <div className="assessment-header">
                    <span className="assessment-icon">{getBiasIcon()}</span>
                    <div className="assessment-text">
                        <h3 style={{ color: getAssessmentColor() }}>
                            {wheel_analysis.exploitable
                                ? 'Ruota Potenzialmente Sfruttabile!'
                                : wheel_analysis.bias_indicators >= 2
                                    ? 'Possibili Anomalie Rilevate'
                                    : 'Ruota Regolare'}
                        </h3>
                        <p>{wheel_analysis.wheel_assessment}</p>
                    </div>
                </div>
                <div className="assessment-stats">
                    <div className="stat-item">
                        <span className="stat-label">Indicatori di Bias</span>
                        <span className="stat-value">{wheel_analysis.bias_indicators}/6</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Spin Analizzati</span>
                        <span className="stat-value">{wheel_analysis.total_spins_analyzed}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Punteggio Clustering</span>
                        <span className="stat-value">{(sector_clustering.cluster_score * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>

            {/* SECTOR CLUSTERING */}
            {!sector_clustering.status && (
                <div className="analysis-section">
                    <h3>📍 Clustering per Settore</h3>
                    <p className="section-desc">
                        Analizza se i numeri si concentrano in zone specifiche della ruota fisica.
                    </p>

                    <div className="cluster-info">
                        <div className="cluster-stat">
                            <span>Distanza Media</span>
                            <span className="value">{sector_clustering.avg_wheel_distance.toFixed(1)}</span>
                        </div>
                        <div className="cluster-stat">
                            <span>Distanza Attesa</span>
                            <span className="value">{sector_clustering.expected_distance.toFixed(1)}</span>
                        </div>
                        <div className="cluster-stat">
                            <span>Bias Rilevato</span>
                            <span className={`value ${sector_clustering.bias_likely ? 'warning' : 'ok'}`}>
                                {sector_clustering.bias_likely ? 'Sì ⚠️' : 'No ✅'}
                            </span>
                        </div>
                    </div>

                    {(sector_clustering.hot_sectors.length > 0 || sector_clustering.cold_sectors.length > 0) && (
                        <div className="sectors-grid">
                            {sector_clustering.hot_sectors.map((s, i) => (
                                <div key={`hot-${i}`} className="sector-badge hot">
                                    🔥 Settore {s.sector + 1}: {s.deviation}
                                </div>
                            ))}
                            {sector_clustering.cold_sectors.map((s, i) => (
                                <div key={`cold-${i}`} className="sector-badge cold">
                                    ❄️ Settore {s.sector + 1}: {s.deviation}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* SUSPICIOUS PAIRS */}
            {pair_analysis.suspicious_pairs && pair_analysis.suspicious_pairs.length > 0 && (
                <div className="analysis-section">
                    <h3>🔗 Coppie Sospette</h3>
                    <p className="section-desc">
                        Numeri che escono consecutivamente più spesso del previsto.
                    </p>

                    <div className="pairs-grid">
                        {pair_analysis.suspicious_pairs.slice(0, 6).map((pair, i) => (
                            <div key={i} className="pair-card">
                                <div className="pair-numbers">
                                    <span
                                        className="number-circle small"
                                        style={{ borderColor: getColorForNumber(pair.from) }}
                                    >
                                        {pair.from}
                                    </span>
                                    <span className="arrow">→</span>
                                    <span
                                        className="number-circle small"
                                        style={{ borderColor: getColorForNumber(pair.to) }}
                                    >
                                        {pair.to}
                                    </span>
                                </div>
                                <div className="pair-info">
                                    <span className="pair-count">{pair.count}x volte</span>
                                    <span className="pair-significance">{pair.significance}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* HOT/COLD ZONES */}
            {!sleeper_anomalies.status && (
                <div className="analysis-section">
                    <h3>🌡️ Zone Calde/Fredde sulla Ruota</h3>
                    <p className="section-desc">
                        Numeri che escono molto di più o molto meno del previsto.
                    </p>

                    <div className="zones-grid">
                        {sleeper_anomalies.hot_numbers.length > 0 && (
                            <div className="zone-card hot">
                                <h4>🔥 Numeri Caldissimi</h4>
                                <p className="zone-desc">Escono più del doppio del previsto</p>
                                <div className="zone-numbers">
                                    {sleeper_anomalies.hot_numbers.map(n => (
                                        <span
                                            key={n.number}
                                            className="number-circle small"
                                            style={{ borderColor: getColorForNumber(n.number) }}
                                        >
                                            {n.number}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {sleeper_anomalies.cold_numbers.length > 0 && (
                            <div className="zone-card cold">
                                <h4>❄️ Numeri Freddi</h4>
                                <p className="zone-desc">Escono meno del 30% del previsto</p>
                                <div className="zone-numbers">
                                    {sleeper_anomalies.cold_numbers.map(n => (
                                        <span
                                            key={n.number}
                                            className="number-circle small"
                                            style={{ borderColor: getColorForNumber(n.number) }}
                                        >
                                            {n.number}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {sleeper_anomalies.wheel_bias_indicator && (
                        <div className="bias-alert">
                            ⚠️ <strong>Attenzione:</strong> I numeri caldi/freddi sono raggruppati sulla ruota fisica - possibile difetto meccanico!
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

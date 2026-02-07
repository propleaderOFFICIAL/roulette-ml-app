import { PatternAnalysisResponse } from '../api';

interface Props {
    data: PatternAnalysisResponse | null;
    loading: boolean;
    error: string | null;
}

const colorMap: Record<string, string> = {
    red: '#ef4444',
    black: '#1f2937',
    green: '#22c55e',
};

const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

export function PatternAnalysis({ data, loading, error }: Props) {
    if (loading) {
        return (
            <div className="card loading-state">
                <div className="spinner"></div>
                <p>Analizzando pattern...</p>
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
                <span className="info-icon">📊</span>
                <p>{data?.error || 'Servono almeno 10 spin per analisi pattern'}</p>
            </div>
        );
    }

    const { patterns } = data;

    return (
        <div className="pattern-analysis">
            <h2 className="section-title">
                <span className="icon">📈</span>
                Analisi Pattern
            </h2>

            {/* Alerts */}
            {patterns.alerts && patterns.alerts.length > 0 && (
                <div className="alerts-section">
                    <h3>🚨 Avvisi Attivi ({patterns.alert_count})</h3>
                    <div className="alerts-list">
                        {patterns.alerts.slice(0, 5).map((alert, idx) => (
                            <div key={idx} className={`alert-item severity-${alert.severity}`}>
                                <span className="alert-icon">
                                    {alert.severity === 'high' ? '🔴' : alert.severity === 'medium' ? '🟡' : '🟢'}
                                </span>
                                <span className="alert-message">{alert.message}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Hot/Cold Numbers */}
            <div className="hot-cold-section">
                <h3>🔥 Numeri Caldi / ❄️ Numeri Freddi</h3>

                <div className="hot-cold-grid">
                    {/* Hot Numbers */}
                    <div className="hot-numbers">
                        <h4>🔥 Caldi</h4>
                        <div className="numbers-row">
                            {patterns.hot_cold.hot.slice(0, 5).map((item) => {
                                const color = item.number === 0 ? 'green'
                                    : RED_NUMBERS.includes(item.number) ? 'red' : 'black';
                                return (
                                    <div key={item.number} className="hot-number-item">
                                        <div
                                            className="number-circle hot"
                                            style={{ borderColor: colorMap[color] }}
                                        >
                                            {item.number}
                                        </div>
                                        <span className="deviation">+{item.deviation.toFixed(1)}σ</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Cold Numbers */}
                    <div className="cold-numbers">
                        <h4>❄️ Freddi</h4>
                        <div className="numbers-row">
                            {patterns.hot_cold.cold.slice(0, 5).map((item) => {
                                const color = item.number === 0 ? 'green'
                                    : RED_NUMBERS.includes(item.number) ? 'red' : 'black';
                                return (
                                    <div key={item.number} className="cold-number-item">
                                        <div
                                            className="number-circle cold"
                                            style={{ borderColor: colorMap[color] }}
                                        >
                                            {item.number}
                                        </div>
                                        <span className="deviation">{item.deviation.toFixed(1)}σ</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>

            {/* Sleeper Numbers */}
            <div className="sleepers-section">
                <h3>💤 Numeri Dormienti</h3>
                <p className="section-desc">Numeri non usciti da molto tempo</p>

                <div className="sleepers-grid">
                    {patterns.sleepers.sleepers.slice(0, 6).map((sleeper) => {
                        const color = sleeper.number === 0 ? 'green'
                            : RED_NUMBERS.includes(sleeper.number) ? 'red' : 'black';
                        return (
                            <div key={sleeper.number} className={`sleeper-item urgency-${sleeper.urgency}`}>
                                <div
                                    className="number-circle"
                                    style={{ backgroundColor: colorMap[color] }}
                                >
                                    {sleeper.number}
                                </div>
                                <div className="sleeper-info">
                                    <span className="gap">{sleeper.gap} spin fa</span>
                                    <span className={`urgency-badge ${sleeper.urgency}`}>
                                        {sleeper.urgency === 'high' ? '⚠️ Molto in ritardo' : '⏰ In ritardo'}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>

                <div className="sleepers-stats">
                    <span>Totale dormienti: {patterns.sleepers.total_sleepers}</span>
                    <span>Gap massimo: {patterns.sleepers.max_gap} spin</span>
                </div>
            </div>

            {/* Streaks */}
            <div className="streaks-section">
                <h3>📊 Serie Attuali</h3>

                <div className="streaks-grid">
                    {Object.entries(patterns.streaks.current_streaks).map(([type, streak]) => (
                        <div key={type} className="streak-item">
                            <span className="streak-type">{getStreakLabel(type)}</span>
                            <div className="streak-value">
                                <span className="streak-count">{streak.length}</span>
                                <span className="streak-label">{streak.value || '-'}</span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Max Streaks */}
                <h4>Record Storici</h4>
                <div className="max-streaks">
                    {Object.entries(patterns.streaks.max_streaks).map(([type, streak]) => (
                        <span key={type} className="max-streak-badge">
                            {getStreakLabel(type)}: {streak.length} ({streak.value})
                        </span>
                    ))}
                </div>
            </div>

            {/* Sector Bias */}
            <div className="sector-bias-section">
                <h3>🎡 Bias Settori Ruota</h3>

                <div className="sectors-grid">
                    {Object.entries(patterns.sector_bias.sectors).map(([sector, sectorData]) => (
                        <div key={sector} className={`sector-item bias-${sectorData.bias_level}`}>
                            <span className="sector-name">{getSectorLabel(sector)}</span>
                            <div className="sector-stats">
                                <span>Atteso: {(sectorData.expected * 100).toFixed(1)}%</span>
                                <span>Attuale: {(sectorData.actual * 100).toFixed(1)}%</span>
                                <span className={`deviation ${sectorData.deviation > 0 ? 'positive' : 'negative'}`}>
                                    {sectorData.deviation > 0 ? '+' : ''}{sectorData.deviation.toFixed(2)}σ
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

                {patterns.sector_bias.bias_detected && (
                    <div className="bias-warning">
                        ⚠️ Rilevato possibile bias nella ruota!
                    </div>
                )}
            </div>
        </div>
    );
}

function getStreakLabel(type: string): string {
    const labels: Record<string, string> = {
        color: '🎨 Colore',
        dozen: '📦 Dozzina',
        column: '📊 Colonna',
        parity: '🔢 Parità',
        high_low: '📈 Alto/Basso',
    };
    return labels[type] || type;
}

function getSectorLabel(sector: string): string {
    const labels: Record<string, string> = {
        voisins: 'Voisins du Zéro',
        tiers: 'Tiers du Cylindre',
        orphelins: 'Orphelins',
    };
    return labels[sector] || sector;
}

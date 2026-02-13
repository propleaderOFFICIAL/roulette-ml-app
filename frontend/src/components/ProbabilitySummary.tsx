import { AdvancedPredictionsResponse, PatternAnalysisResponse } from '../api';
import './ProbabilitySummary.css';

interface Props {
    data: AdvancedPredictionsResponse | null;
    patternData: PatternAnalysisResponse | null;
    loading: boolean;
}

const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

// ── Thresholds per category ──
// Each category has a minimum confidence AND minimum agreement to appear
// Agreement 0.8 means at least 4 out of 5 AI models must agree
const THRESHOLDS: Record<string, { minConf: number; minAgree: number }> = {
    color: { minConf: 0.60, minAgree: 0.80 },
    parity: { minConf: 0.60, minAgree: 0.80 },
    high_low: { minConf: 0.60, minAgree: 0.80 },
    dozen: { minConf: 0.45, minAgree: 0.80 },
    column: { minConf: 0.45, minAgree: 0.80 },
    sector: { minConf: 0.55, minAgree: 0.80 },
    number: { minConf: 0.05, minAgree: 0.80 },
};

interface SummaryCard {
    category: string;       // internal key
    icon: string;
    label: string;          // display name
    value: string;          // predicted outcome
    confidence: number;     // probability of the outcome
    agreement: number;      // 0-1, fraction of models agreeing
    agreeingModels: number;
    totalModels: number;
    accentColor: string;    // border & glow color
}

export function ProbabilitySummary({ data, patternData, loading }: Props) {
    if (loading) {
        return (
            <div className="riepilogo-container">
                <div className="loading-state">
                    <div className="spinner"></div>
                    <p>Analisi probabilità in corso...</p>
                </div>
            </div>
        );
    }

    if (!data || data.error) {
        return (
            <div className="riepilogo-container">
                <div className="info-state">
                    <p>⚠️ {data?.error || 'Dati non disponibili. Inserisci più spin.'}</p>
                </div>
            </div>
        );
    }

    const cards = buildCards(data);

    return (
        <div className="riepilogo-container">
            <div className="riepilogo-header">
                <h2 className="riepilogo-title">
                    <span className="icon">💎</span> Riepilogo Previsioni
                </h2>
                <span className="riepilogo-badge">Solo alta probabilità</span>
            </div>

            {cards.length > 0 ? (
                <div className="riepilogo-grid">
                    {cards.map((card, idx) => {
                        const numVal = card.category === 'number' ? parseInt(card.value) : null;
                        const numColor = numVal !== null
                            ? (numVal === 0 ? '#22c55e' : RED_NUMBERS.includes(numVal) ? '#ef4444' : '#a0aec0')
                            : undefined;

                        return (
                            <div
                                key={`${card.category}-${idx}`}
                                className="riepilogo-card"
                                style={{ '--card-accent': card.accentColor } as React.CSSProperties}
                            >
                                {/* Category label */}
                                <div className="rc-category">
                                    <span className="rc-icon">{card.icon}</span>
                                    <span className="rc-label">{card.label}</span>
                                </div>

                                {/* Predicted value */}
                                <div
                                    className="rc-value"
                                    style={numColor ? { color: numColor, textShadow: `0 0 12px ${numColor}50` } : undefined}
                                >
                                    {card.value}
                                </div>

                                {/* Confidence */}
                                <div className="rc-confidence">
                                    {(card.confidence * 100).toFixed(1)}%
                                </div>

                                {/* Agreement bar */}
                                <div className="rc-agreement">
                                    <div className="rc-agree-bar">
                                        <div
                                            className="rc-agree-fill"
                                            style={{ width: `${card.agreement * 100}%` }}
                                        />
                                    </div>
                                    <span className="rc-agree-text">
                                        {card.agreeingModels}/{card.totalModels} AI
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="riepilogo-empty">
                    <span className="icon">⚖️</span>
                    <h3>Nessuna previsione rilevante</h3>
                    <p>Le AI non hanno trovato previsioni con probabilità e accordo sufficienti. Inserisci più spin.</p>
                </div>
            )}
        </div>
    );
}

// ── Build the unified card list ──

function buildCards(data: AdvancedPredictionsResponse): SummaryCard[] {
    const cards: SummaryCard[] = [];
    const totalModels = data.number ? Object.keys(data.number.models).length : 5;

    // 1. Color
    if (data.color) {
        const entries = Object.entries(data.color.ensemble).sort((a, b) => b[1] - a[1]);
        if (entries.length > 0) {
            const [color, prob] = entries[0];
            const t = THRESHOLDS.color;
            if (prob >= t.minConf && data.color.agreement >= t.minAgree) {
                cards.push({
                    category: 'color',
                    icon: '🎨',
                    label: 'Colore',
                    value: color === 'red' ? 'Rosso 🔴' : 'Nero ⚫',
                    confidence: prob,
                    agreement: data.color.agreement,
                    agreeingModels: Math.round(data.color.agreement * totalModels),
                    totalModels,
                    accentColor: color === 'red' ? '#ef4444' : '#a0aec0',
                });
            }
        }
    }

    // 2. Number
    if (data.number && data.number.ensemble.length > 0) {
        const top = data.number.ensemble[0];
        const t = THRESHOLDS.number;
        if (top.probability >= t.minConf && data.number.agreement >= t.minAgree) {
            cards.push({
                category: 'number',
                icon: '🔢',
                label: 'Numero',
                value: top.number.toString(),
                confidence: top.probability,
                agreement: data.number.agreement,
                agreeingModels: Math.round(data.number.agreement * totalModels),
                totalModels,
                accentColor: '#8b5cf6',
            });
        }
    }

    // 3. Betting areas
    if (data.betting_areas) {
        const areaConfigs: {
            key: keyof typeof data.betting_areas;
            thresholdKey: string;
            icon: string;
            label: string;
            accent: string;
            formatValue?: (v: string) => string;
        }[] = [
                {
                    key: 'dozen', thresholdKey: 'dozen', icon: '🎲', label: 'Dozzina',
                    accent: '#06b6d4',
                    formatValue: (v) => {
                        if (v.includes('1')) return '1ª (1-12)';
                        if (v.includes('2')) return '2ª (13-24)';
                        return '3ª (25-36)';
                    }
                },
                {
                    key: 'column', thresholdKey: 'column', icon: '📊', label: 'Colonna',
                    accent: '#a78bfa',
                    formatValue: (v) => {
                        if (v.includes('1')) return '1ª Col';
                        if (v.includes('2')) return '2ª Col';
                        return '3ª Col';
                    }
                },
                {
                    key: 'high_low', thresholdKey: 'high_low', icon: '↕️', label: 'Alto/Basso',
                    accent: '#f472b6',
                    formatValue: (v) => v.includes('Alto') || v.includes('19') ? 'Alto (19-36)' : 'Basso (1-18)'
                },
                {
                    key: 'parity', thresholdKey: 'parity', icon: '🔄', label: 'Pari/Dispari',
                    accent: '#818cf8',
                    formatValue: (v) => v.includes('Pari') || v === 'even' ? 'Pari' : 'Dispari'
                },
                {
                    key: 'sector', thresholdKey: 'sector', icon: '🎡', label: 'Settore',
                    accent: '#f59e0b',
                },
            ];

        for (const cfg of areaConfigs) {
            const area = data.betting_areas[cfg.key];
            if (!area || typeof area !== 'object' || !('agreement' in area)) continue;

            const t = THRESHOLDS[cfg.thresholdKey];
            if (area.confidence >= t.minConf && area.agreement >= t.minAgree) {
                const displayValue = cfg.formatValue ? cfg.formatValue(area.prediction) : area.prediction;
                cards.push({
                    category: cfg.thresholdKey,
                    icon: cfg.icon,
                    label: cfg.label,
                    value: displayValue,
                    confidence: area.confidence,
                    agreement: area.agreement,
                    agreeingModels: Math.round(area.agreement * totalModels),
                    totalModels,
                    accentColor: cfg.accent,
                });
            }
        }
    }

    // Sort by confidence descending
    cards.sort((a, b) => b.confidence - a.confidence);
    return cards;
}

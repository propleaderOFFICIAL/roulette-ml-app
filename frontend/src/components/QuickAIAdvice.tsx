import { AdvancedPredictionsResponse, PatternAnalysisResponse, WheelClusteringResponse } from '../api';
import './QuickAIAdvice.css';

interface Props {
    predictions: AdvancedPredictionsResponse | null;
    patterns: PatternAnalysisResponse | null;
    wheelAnalysis: WheelClusteringResponse | null;
    loading: boolean;
}

const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

export function QuickAIAdvice({ predictions, patterns, wheelAnalysis, loading }: Props) {
    if (loading) {
        return (
            <div className="quick-ai-container">
                <div className="quick-ai-loading">
                    <div className="spinner small"></div>
                    <span>Elaborazione consigli IA...</span>
                </div>
            </div>
        );
    }

    // Collect all advice from different sources
    const topAdvice = getTopAdvice(predictions, patterns, wheelAnalysis);
    const bestBets = getBestBets(predictions, patterns, wheelAnalysis);

    if (topAdvice.length === 0 && bestBets.length === 0) {
        return (
            <div className="quick-ai-container">
                <div className="quick-ai-empty">
                    <span className="icon">🎰</span>
                    <span>Inserisci spin per ricevere consigli AI</span>
                </div>
            </div>
        );
    }

    return (
        <div className="quick-ai-container">
            {/* PILLOLA PRINCIPALE - IL CONSIGLIO PIÙ IMPORTANTE */}
            {topAdvice.length > 0 && (
                <div className={`main-advice-pill urgency-${topAdvice[0].urgency}`}>
                    <div className="pill-icon">{topAdvice[0].icon}</div>
                    <div className="pill-content">
                        <span className="pill-title">{topAdvice[0].title}</span>
                        <span className="pill-action">{topAdvice[0].action}</span>
                    </div>
                    <div className="pill-urgency-badge">{topAdvice[0].urgencyLabel}</div>
                </div>
            )}

            {/* BETS PILLS - SCOMMESSE CONSIGLIATE */}
            {bestBets.length > 0 && (
                <div className="best-bets-row">
                    <span className="bets-label">🎯 Suggerimenti:</span>
                    <div className="bets-pills">
                        {bestBets.slice(0, 4).map((bet, idx) => (
                            <div key={idx} className={`bet-pill ${bet.type}`}>
                                {bet.icon && <span className="bet-icon">{bet.icon}</span>}
                                <span className="bet-text">{bet.text}</span>
                                {bet.prob && <span className="bet-prob">{bet.prob}</span>}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ALTRI CONSIGLI URGENTI (se presenti) */}
            {topAdvice.length > 1 && (
                <div className="secondary-advice-row">
                    {topAdvice.slice(1, 3).map((advice, idx) => (
                        <div key={idx} className={`secondary-advice-card urgency-${advice.urgency}`}>
                            <span className="adv-icon">{advice.icon}</span>
                            <div className="adv-content">
                                <span className="adv-title">{advice.title}</span>
                                <span className="adv-action">{advice.action}</span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

interface AdviceItem {
    urgency: 'high' | 'medium' | 'low';
    urgencyLabel: string;
    icon: string;
    title: string;
    action: string;
    score: number; // for sorting
}

interface BetItem {
    type: 'number' | 'color' | 'sector' | 'pattern';
    icon?: string;
    text: string;
    prob?: string;
}

function getTopAdvice(
    _predictions: AdvancedPredictionsResponse | null,
    patterns: PatternAnalysisResponse | null,
    wheelAnalysis: WheelClusteringResponse | null
): AdviceItem[] {
    const advice: AdviceItem[] = [];

    // 1. Check for wheel bias (highest priority if exploitable)
    if (wheelAnalysis?.wheel_analysis?.exploitable) {
        advice.push({
            urgency: 'high',
            urgencyLabel: '🔥 URGENTE',
            icon: '🎡',
            title: 'Ruota Sbilanciata Rilevata!',
            action: 'Settori specifici escono più del previsto. Vedi analisi ruota sotto.',
            score: 100
        });
    }

    // 2. Check for sleepers (high urgency)
    const topSleeper = patterns?.patterns?.sleepers?.sleepers?.[0];
    if (topSleeper && topSleeper.urgency === 'high') {
        advice.push({
            urgency: 'high',
            urgencyLabel: '⏰ RITARDO',
            icon: '💤',
            title: `Il ${topSleeper.number} non esce da ${topSleeper.gap} giri`,
            action: `Pieno sul ${topSleeper.number}`,
            score: 90
        });
    }

    // 3. Check for color streaks
    const colorStreak = patterns?.patterns?.streaks?.current_streaks?.['color'];
    if (colorStreak && colorStreak.length >= 4) {
        const isRed = colorStreak.value === 'red';
        advice.push({
            urgency: 'medium',
            urgencyLabel: '🌊 SERIE',
            icon: isRed ? '🔴' : '⚫',
            title: `${colorStreak.length} ${isRed ? 'Rossi' : 'Neri'} di fila`,
            action: `Punta sul ${isRed ? 'Nero' : 'Rosso'}`,
            score: 70 + colorStreak.length
        });
    }

    // 4. Check for hot sectors
    if (patterns?.patterns?.sector_bias?.sectors) {
        const sectors = patterns.patterns.sector_bias.sectors;
        for (const [name, data] of Object.entries(sectors)) {
            if (data.deviation > 2.5) {
                advice.push({
                    urgency: 'medium',
                    urgencyLabel: '🔥 CALDO',
                    icon: '🎡',
                    title: `${getSectorName(name)} molto caldo`,
                    action: `Punta su ${getSectorName(name)}`,
                    score: 60 + data.deviation * 5
                });
            }
        }
    }

    // 5. Cold wheel zones from wheel analysis
    if (wheelAnalysis?.wheel_analysis?.sleeper_anomalies?.wheel_bias_indicator) {
        advice.push({
            urgency: 'medium',
            urgencyLabel: '⚡ ANOMALIA',
            icon: '🌡️',
            title: 'Zone calde/fredde raggruppate',
            action: 'La ruota mostra pattern fisici. Vedi analisi.',
            score: 65
        });
    }

    // Sort by score and return top 3
    return advice.sort((a, b) => b.score - a.score).slice(0, 3);
}

function getBestBets(
    predictions: AdvancedPredictionsResponse | null,
    _patterns: PatternAnalysisResponse | null,
    wheelAnalysis: WheelClusteringResponse | null
): BetItem[] {
    const bets: BetItem[] = [];

    // 1. Top predicted number
    if (predictions?.number?.ensemble?.[0]) {
        const topNum = predictions.number.ensemble[0];
        const color = topNum.number === 0 ? 'green'
            : RED_NUMBERS.includes(topNum.number) ? 'red' : 'black';
        bets.push({
            type: 'number',
            icon: color === 'red' ? '🔴' : color === 'black' ? '⚫' : '🟢',
            text: `${topNum.number}`,
            prob: `${(topNum.probability * 100).toFixed(1)}%`
        });
    }

    // 2. Predicted color
    if (predictions?.color?.ensemble) {
        const colors = predictions.color.ensemble;
        const sortedColors = Object.entries(colors).sort(([, a], [, b]) => b - a);
        if (sortedColors[0]) {
            const [colorName, prob] = sortedColors[0];
            bets.push({
                type: 'color',
                icon: colorName === 'red' ? '🔴' : colorName === 'black' ? '⚫' : '🟢',
                text: colorName.charAt(0).toUpperCase() + colorName.slice(1),
                prob: `${(prob * 100).toFixed(0)}%`
            });
        }
    }

    // 3. Best betting area
    if (predictions?.betting_areas) {
        const areas = predictions.betting_areas;
        let bestArea = { name: '', prob: 0 };

        // Check dozens
        const dozenProbs = areas.dozen?.probabilities || {};
        for (const [name, prob] of Object.entries(dozenProbs)) {
            if (prob > bestArea.prob) {
                bestArea = { name: `${name}`, prob };
            }
        }

        // Check high/low
        const hlProbs = areas.high_low?.probabilities || {};
        for (const [name, prob] of Object.entries(hlProbs)) {
            if (prob > bestArea.prob) {
                bestArea = { name: name === 'high' ? 'Alti 19-36' : 'Bassi 1-18', prob };
            }
        }

        if (bestArea.prob > 0.35) {
            bets.push({
                type: 'pattern',
                text: bestArea.name,
                prob: `${(bestArea.prob * 100).toFixed(0)}%`
            });
        }
    }

    // 4. Hot sector from wheel analysis
    if (wheelAnalysis?.wheel_analysis?.sector_clustering?.hot_sectors?.length) {
        const hotSector = wheelAnalysis.wheel_analysis.sector_clustering.hot_sectors[0];
        bets.push({
            type: 'sector',
            icon: '🎡',
            text: `Settore ${hotSector.sector + 1}`,
            prob: hotSector.deviation
        });
    }

    return bets;
}

function getSectorName(sector: string): string {
    const names: Record<string, string> = {
        voisins: 'Vicini',
        tiers: 'Tiers',
        orphelins: 'Orfanelli'
    };
    return names[sector] || sector;
}

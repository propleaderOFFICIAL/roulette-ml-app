import React, { useMemo } from 'react';
import { AdvancedPredictionsResponse } from '../api';
import './RouletteTableVisualization.css';

interface Props {
    data: AdvancedPredictionsResponse | null;
}

const RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

export function RouletteTableVisualization({ data }: Props) {
    // Flatten probabilities for easy access
    const numberProbs = useMemo(() => {
        if (!data?.number?.ensemble) return {};
        const map: Record<number, number> = {};
        // Initialize with 0
        for (let i = 0; i <= 36; i++) map[i] = 0;

        data.number.ensemble.forEach(item => {
            map[item.number] = item.probability;
        });
        return map;
    }, [data]);

    const maxProb = Math.max(...Object.values(numberProbs), 0.001); // Avoid div by zero

    const getNumberStyle = (num: number) => {
        const isRed = RED_NUMBERS.includes(num);
        const isZero = num === 0;
        const prob = numberProbs[num] || 0;
        const intensity = prob / maxProb; // 0 to 1 relative to max

        // Base color
        let backgroundColor = isZero ? '#22c55e' : isRed ? '#ef4444' : '#1f2937';

        // Highlight based on probability (glow or border)
        const style: React.CSSProperties = {
            backgroundColor,
            opacity: 0.8 + (intensity * 0.2), // Slight opacity variation
        };

        if (intensity > 0.5) {
            style.boxShadow = `0 0 ${10 + intensity * 10}px ${intensity * 4}px rgba(255, 255, 0, ${intensity})`;
            style.zIndex = 10;
            style.transform = 'scale(1.05)';
        }

        return style;
    };

    // Specific helpers for different bet types to normalize intensity
    const getStyle = (prob: number, thresholdHigh: number, thresholdVeryHigh: number, baseColor?: string) => {
        const style: React.CSSProperties & { [key: string]: any } = {
            border: '2px solid rgba(255,255,255,0.2)',
            '--scale': 1,
        };

        if (baseColor) style.backgroundColor = baseColor;

        if (prob >= thresholdHigh) {
            style.boxShadow = `inset 0 0 20px rgba(255, 215, 0, 0.3)`; // Gold glow
            style.border = '2px solid rgba(255, 215, 0, 0.6)';
        }

        if (prob >= thresholdVeryHigh) {
            style.boxShadow = `inset 0 0 40px rgba(255, 215, 0, 0.5), 0 0 10px rgba(255, 215, 0, 0.3)`;
            style.border = '2px solid rgba(255, 215, 0, 1)';
            style['--scale'] = 1.02;
            style.zIndex = 5;
            style.fontWeight = 'bold';
        }
        return style;
    }

    if (!data || data.error) {
        return (
            <div className="card info-state">
                <p>⚠️ {data?.error || 'Dati insufficienti per la visualizzazione (servono ~30 spin).'}</p>
            </div>
        );
    }

    // Safely access data
    const cols = data.betting_areas?.column.probabilities || {};
    const dozens = data.betting_areas?.dozen.probabilities || {};
    const parity = data.betting_areas?.parity.probabilities || {};
    const highlow = data.betting_areas?.high_low.probabilities || {};
    const colors = data.color?.ensemble || {};

    return (
        <div className="roulette-table-container">
            <div className="roulette-grid">
                {/* 0 */}
                <div className="roulette-cell zero" style={getNumberStyle(0)}>
                    <span className="number-label">0</span>
                    <span className="prob-label">{(numberProbs[0] * 100).toFixed(1)}%</span>
                </div>

                {/* 1-36 Numbers */}
                <div className="numbers-grid-layout">
                    {Array.from({ length: 12 }).map((_, rowIndex) => {
                        // Rows are 1,2,3 then 4,5,6...
                        // But standard board layout is often:
                        // 3 6 9 ...
                        // 2 5 8 ...
                        // 1 4 7 ...
                        // when viewed sideways. The user image shows:
                        // 1 2 3
                        // 4 5 6 ...
                        // Let's stick to numerical order for simplicity or column-based?
                        // Standard table has columns:
                        // Col 1: 1, 4, 7, ... 34
                        // Col 2: 2, 5, 8, ... 35
                        // Col 3: 3, 6, 9, ... 36
                        // Let's implement this standard layout (3 rows, 12 cols) if referring to the long side.
                        // Or 12 rows, 3 cols if vertical.
                        // User image is horizontal: 3 rows of numbers.
                        // Row 1 (top): 3, 6, 9, ... 36
                        // Row 2 (mid): 2, 5, 8, ... 35
                        // Row 3 (bot): 1, 4, 7, ... 34

                        const colIndex = rowIndex; // 0 to 11
                        const n1 = 3 + colIndex * 3; // Top row number (3, 6...)
                        const n2 = 2 + colIndex * 3; // Mid row number (2, 5...)
                        const n3 = 1 + colIndex * 3; // Bot row number (1, 4...)

                        return (
                            <div key={rowIndex} className="table-column">
                                <div className="roulette-cell" style={getNumberStyle(n1)}>
                                    <span className="number-label">{n1}</span>
                                    <span className="prob-label">{(numberProbs[n1] * 100).toFixed(1)}%</span>
                                </div>
                                <div className="roulette-cell" style={getNumberStyle(n2)}>
                                    <span className="number-label">{n2}</span>
                                    <span className="prob-label">{(numberProbs[n2] * 100).toFixed(1)}%</span>
                                </div>
                                <div className="roulette-cell" style={getNumberStyle(n3)}>
                                    <span className="number-label">{n3}</span>
                                    <span className="prob-label">{(numberProbs[n3] * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* 2 to 1 Columns */}
                <div className="column-bets">
                    <div className="roulette-cell bet-area" style={getStyle(cols['3ª col'] || 0, 0.35, 0.45)}>
                        <span>2 to 1</span>
                        <span className="prob-label-mini">{((cols['3ª col'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(cols['2ª col'] || 0, 0.35, 0.45)}>
                        <span>2 to 1</span>
                        <span className="prob-label-mini">{((cols['2ª col'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(cols['1ª col'] || 0, 0.35, 0.45)}>
                        <span>2 to 1</span>
                        <span className="prob-label-mini">{((cols['1ª col'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                </div>

                {/* Dozens */}
                <div className="dozens-bets">
                    <div className="roulette-cell bet-area" style={getStyle(dozens['1ª (1-12)'] || 0, 0.35, 0.45)}>
                        <span>1st 12</span>
                        <span className="prob-label-mini">{((dozens['1ª (1-12)'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(dozens['2ª (13-24)'] || 0, 0.35, 0.45)}>
                        <span>2nd 12</span>
                        <span className="prob-label-mini">{((dozens['2ª (13-24)'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(dozens['3ª (25-36)'] || 0, 0.35, 0.45)}>
                        <span>3rd 12</span>
                        <span className="prob-label-mini">{((dozens['3ª (25-36)'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                </div>

                {/* Simple Bets */}
                <div className="simple-bets">
                    <div className="roulette-cell bet-area" style={getStyle(highlow['Basso (1-18)'] || 0, 0.52, 0.6)}>
                        <span>1-18</span>
                        <span className="prob-label-mini">{((highlow['Basso (1-18)'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(parity['Pari'] || 0, 0.52, 0.6)}>
                        <span>EVEN</span>
                        <span className="prob-label-mini">{((parity['Pari'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area red" style={getStyle(colors['red'] || 0, 0.52, 0.6, '#ef4444')}>
                        <span>RED</span>
                        <span className="prob-label-mini">{((colors['red'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area black" style={getStyle(colors['black'] || 0, 0.52, 0.6, '#1f2937')}>
                        <span>BLACK</span>
                        <span className="prob-label-mini">{((colors['black'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(parity['Dispari'] || 0, 0.52, 0.6)}>
                        <span>ODD</span>
                        <span className="prob-label-mini">{((parity['Dispari'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="roulette-cell bet-area" style={getStyle(highlow['Alto (19-36)'] || 0, 0.52, 0.6)}>
                        <span>19-36</span>
                        <span className="prob-label-mini">{((highlow['Alto (19-36)'] || 0) * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>

            <div className="table-legend">
                <p>💡 Le aree evidenziate in oro indicano un'alta probabilità di vincita.</p>
            </div>
        </div>
    );
}

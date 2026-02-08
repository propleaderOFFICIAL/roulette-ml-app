import { PatternAnalysisResponse, PatternAnalysis as PatternType } from '../api';
import './PatternAnalysis.css';

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
                <p>Sto analizzando la ruota...</p>
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
                <p>📊 {data?.error || 'In attesa di abbastanza dati (minimo 10 spin)...'}</p>
            </div>
        );
    }

    const { patterns } = data;
    const advice = getBettingAdvice(patterns);

    return (
        <div className="pattern-analysis-container">
            <h2 className="section-title">
                <span className="icon">🧠</span> Strategia & Consigli IA
            </h2>

            {/* SEZIONE 1: I CONSIGLI DELL'IA */}
            <div className="advice-section">
                {advice.length > 0 ? (
                    <div className="advice-grid">
                        {advice.map((item, idx) => (
                            <div key={idx} className={`advice-card severity-${item.severity}`}>
                                <div className="advice-header">
                                    <span className="advice-icon">{item.icon}</span>
                                    <span className="advice-title">{item.title}</span>
                                </div>
                                <p className="advice-desc">{item.description}</p>
                                <div className="advice-action">
                                    <strong>Il Consiglio:</strong> {item.action}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="no-advice">
                        <p>✅ Situazione bilanciata. Nessuna anomalia statistica rilevante al momento.</p>
                    </div>
                )}
            </div>

            {/* SEZIONE 2: CLIMA DEL TAVOLO (Caldi/Freddi/Ritardatari) */}
            <div className="table-climate-section">

                {/* Numeri Caldi */}
                <div className="climate-group">
                    <h3>🔥 Numeri in Fiamme</h3>
                    <p className="climate-sub">Stanno uscendo molto spesso</p>
                    <div className="numbers-row">
                        {patterns.hot_cold.hot.slice(0, 5).map((item) => {
                            const color = item.number === 0 ? 'green'
                                : RED_NUMBERS.includes(item.number) ? 'red' : 'black';
                            return (
                                <div key={item.number} className="climate-number">
                                    <div className="number-circle" style={{ borderColor: colorMap[color], boxShadow: `0 0 10px ${colorMap[color]}` }}>
                                        {item.number}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>

                {/* Ritardatari (Sleepers) */}
                <div className="climate-group">
                    <h3>💤 I Grandi Assenti</h3>
                    <p className="climate-sub">Non escono da molto tempo</p>
                    <div className="numbers-row">
                        {patterns.sleepers.sleepers.slice(0, 5).map((sleeper) => {
                            const color = sleeper.number === 0 ? 'green'
                                : RED_NUMBERS.includes(sleeper.number) ? 'red' : 'black';
                            return (
                                <div key={sleeper.number} className="climate-number">
                                    <div className="number-circle sleeper" style={{ backgroundColor: colorMap[color], opacity: 0.7 }}>
                                        {sleeper.number}
                                    </div>
                                    <span className="gap-label">{sleeper.gap} giri fa</span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </div>

            {/* SEZIONE 3: TREND ATTUALI */}
            <div className="trends-section">
                <h3>🌊 Onde e Serie Attuali</h3>
                <div className="trends-grid">
                    {Object.entries(patterns.streaks.current_streaks).map(([type, streak]) => {
                        if (streak.length < 2) return null; // Nascondi serie irrilevanti
                        const label = getFriendlyStreakLabel(type, streak.value);
                        return (
                            <div key={type} className="trend-item">
                                <span className="trend-name">{label}</span>
                                <div className="trend-bar-container">
                                    <div className="trend-bar" style={{ width: `${Math.min(streak.length * 10, 100)}%` }}>
                                        {streak.length} di fila
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
                {Object.values(patterns.streaks.current_streaks).every(s => s.length < 2) && (
                    <p className="no-trends">Il tavolo è molto caotico, non ci sono serie dominanti al momento.</p>
                )}
            </div>

            {/* SEZIONE 4: SCHEMI & SETTORI DA GIOCARE */}
            <div className="schemes-section">
                <h3>🎯 Schemi & Settori Caldi</h3>
                <p className="section-desc">Settori della ruota e pattern speciali identificati dall'IA.</p>

                <div className="schemes-grid">
                    {/* Settori (Voisins, Tiers, Orphelins) */}
                    {Object.entries(patterns.sector_bias.sectors).map(([sector, data]) => {
                        const isHot = data.deviation > 1.0;
                        const isCold = data.deviation < -1.0;
                        if (!isHot && !isCold && Math.abs(data.deviation) < 0.5) return null; // Nascondi neutri

                        return (
                            <div key={sector} className={`scheme-card ${isHot ? 'hot' : 'cold'}`}>
                                <div className="scheme-header">
                                    <span className="scheme-icon">🎡</span>
                                    <div className="scheme-title-group">
                                        <span className="scheme-title">{getSectorLabel(sector)}</span>
                                        <span className="scheme-sub">{getSectornumbers(sector)}</span>
                                    </div>
                                </div>
                                <p className="scheme-desc">{getSectorExplanation(sector)}</p>
                                <div className="scheme-numbers-list">
                                    <strong>Numeri:</strong> {getSectorSpecificNumbers(sector)}
                                </div>
                                <div className="scheme-how-to-bet">
                                    <strong>Come Puntare:</strong> {getSectorHowToBet(sector)}
                                </div>
                                <div className="scheme-stat">
                                    <span className="scheme-label">🔥 Intensità Uscite</span>
                                    <span className="scheme-value">{(data.actual * 100).toFixed(0)}%</span>
                                </div>
                                <div className="scheme-action">
                                    {isHot ? '🟢 Ottimo da giocare ora' : '❄️ Freddo - Non giocare'}
                                </div>
                            </div>
                        );
                    })}

                    {/* Pattern Alerts Generici */}
                    {patterns.alerts.slice(0, 3).map((alert, idx) => (
                        <div key={idx} className={`scheme-card alert severity-${alert.severity}`}>
                            <div className="scheme-header">
                                <span className="scheme-icon">⚡</span>
                                <span className="scheme-title">Pattern Rilevato</span>
                            </div>
                            <p className="scheme-desc">{translateAlertMessage(alert.message)}</p>
                            <div className="scheme-action">
                                💡 {getPatternAdvice(alert.message)}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Fallback se non ci sono schemi */}
                {Object.values(patterns.sector_bias.sectors).every(d => Math.abs(d.deviation) < 1.0) && patterns.alerts.length === 0 && (
                    <div className="no-schemes">
                        <p>Nessun settore o schema particolare da segnalare al momento.</p>
                    </div>
                )}
            </div>

        </div>
    );
}

// --- LOGICA DI GENERAZIONE CONSIGLI ---

interface AdviceItem {
    severity: 'high' | 'medium' | 'low';
    icon: string;
    title: string;
    description: string;
    action: string;
}

function getBettingAdvice(patterns: PatternType): AdviceItem[] {
    const advice: AdviceItem[] = [];

    // 1. Analisi Ritardatari (Sleepers)
    const topSleeper = patterns.sleepers.sleepers[0];
    if (topSleeper && topSleeper.urgency === 'high') {
        advice.push({
            severity: 'high',
            icon: '⏰',
            title: `Il Numero ${topSleeper.number} Non Esce Da Molto`,
            description: `Non esce da ${topSleeper.gap} giri. Più tempo passa, più cresce la "pressione statistica".`,
            action: `Metti una piccola fiche direttamente sul numero ${topSleeper.number} (puntata "pieno" = fiche al centro del numero).`
        });
    }

    // 2. Analisi Serie Colore (Streaks)
    const colorStreak = patterns.streaks.current_streaks['color'];
    if (colorStreak && colorStreak.length >= 4) {
        const isRed = colorStreak.value === 'red';
        const colorName = isRed ? 'Rosso' : 'Nero';
        const oppositeColor = isRed ? 'Nero' : 'Rosso';

        advice.push({
            severity: 'medium',
            icon: '🎨',
            title: `${colorStreak.length} ${colorName} di Fila`,
            description: `È uscito ${colorName} per ${colorStreak.length} volte consecutive. Le serie lunghe prima o poi si interrompono.`,
            action: `Punta sul ${oppositeColor} (il diamante ${isRed ? 'nero' : 'rosso'} sul tavolo).`
        });
    }

    // 3. Bias Settori (Se un settore è molto caldo)
    const tiers = patterns.sector_bias.sectors['tiers'];
    if (tiers && tiers.deviation > 2.0) {
        advice.push({
            severity: 'medium',
            icon: '🎡',
            title: 'Zona "Terzi" Molto Calda',
            description: 'I numeri 5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36 stanno uscendo spesso. Sono i numeri sul lato opposto allo zero sulla ruota.',
            action: '🎰 Online: cerca bottone "Tiers". 🎲 Dal vivo: dì "Tiers" al croupier.'
        });
    }

    const voisins = patterns.sector_bias.sectors['voisins'];
    if (voisins && voisins.deviation > 2.0) {
        advice.push({
            severity: 'medium',
            icon: '🎡',
            title: 'Zona "Vicini dello Zero" Calda',
            description: 'I numeri attorno allo zero sulla ruota stanno uscendo spesso (0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35).',
            action: '🎰 Online: cerca bottone "Voisins" o "Vicini". 🎲 Dal vivo: dì "Vicini dello zero" al croupier.'
        });
    }

    // 4. Analisi Dozzine
    const dozenStreak = patterns.streaks.current_streaks['dozen'];
    if (dozenStreak && dozenStreak.length >= 3) {
        const dozenName = dozenStreak.value === 'first' ? '1ª Dozzina (1-12)' :
            dozenStreak.value === 'second' ? '2ª Dozzina (13-24)' : '3ª Dozzina (25-36)';
        advice.push({
            severity: 'low',
            icon: '📦',
            title: `${dozenStreak.length} Uscite nella Stessa Dozzina`,
            description: `La ${dozenName} è uscita ${dozenStreak.length} volte di fila. È raro che continui oltre 4-5 volte.`,
            action: `Valuta di puntare sulle altre due dozzine (le caselle "1st 12", "2nd 12", "3rd 12" sul tavolo).`
        });
    }

    return advice.slice(0, 3);
}

function getFriendlyStreakLabel(type: string, value: string | null): string {
    if (type === 'color') return value === 'red' ? '🔴 Serie Rossi' : '⚫️ Serie Neri';
    if (type === 'parity') return value === 'even' ? '🔢 Serie Pari' : '🔢 Serie Dispari';
    if (type === 'high_low') return value === 'high' ? '📈 Serie Alti (19-36)' : '📉 Serie Bassi (1-18)';
    if (type === 'dozen') return '📦 Stessa Dozzina';
    if (type === 'column') return '📊 Stessa Colonna';
    return type;
}

function getSectorLabel(sector: string): string {
    const labels: Record<string, string> = {
        voisins: 'Vicini dello Zero',
        tiers: 'Terzi (Lato Opposto)',
        orphelins: 'Orfanelli',
    };
    return labels[sector] || sector;
}

function getSectornumbers(sector: string): string {
    if (sector === 'voisins') return '17 numeri attorno allo 0 sulla ruota';
    if (sector === 'tiers') return '12 numeri sul lato opposto allo 0';
    if (sector === 'orphelins') return '8 numeri rimanenti';
    return '';
}

function getSectorExplanation(sector: string): string {
    if (sector === 'voisins') return 'Sulla ruota fisica, sono i numeri che stanno "vicini" allo zero. Quando la pallina cade in quella zona, esce uno di questi.';
    if (sector === 'tiers') return 'Sono i numeri che stanno dalla parte opposta allo zero sulla ruota fisica.';
    if (sector === 'orphelins') return 'Sono gli 8 numeri che non fanno parte degli altri due gruppi.';
    return '';
}

function getSectorHowToBet(sector: string): string {
    if (sector === 'voisins') return '🎰 Online: cerca il bottone "Voisins" o "Vicini". 🎲 Dal vivo: dì al croupier "Vicini dello zero, X euro" e lui piazza per te.';
    if (sector === 'tiers') return '🎰 Online: cerca il bottone "Tiers" o "Serie 5/8". 🎲 Dal vivo: dì "Tiers, X euro" al croupier.';
    if (sector === 'orphelins') return '🎰 Online: cerca il bottone "Orphelins". 🎲 Dal vivo: dì "Orfanelli, X euro" al croupier.';
    return '';
}

function getSectorSpecificNumbers(sector: string): string {
    if (sector === 'voisins') return '0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35';
    if (sector === 'tiers') return '5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36';
    if (sector === 'orphelins') return '1, 6, 9, 14, 17, 20, 31, 34';
    return '';
}

function translateAlertMessage(msg: string): string {
    const lMsg = msg.toLowerCase();

    // Sector frequency patterns
    if (lMsg.includes('voisins') && lMsg.includes('moderate low frequency'))
        return 'Voisins sta uscendo poco (freq. medio-bassa).';
    if (lMsg.includes('voisins') && lMsg.includes('low frequency'))
        return 'Voisins è freddo (freq. bassa).';
    if (lMsg.includes('tiers') && lMsg.includes('moderate low frequency'))
        return 'Tiers sta uscendo poco (freq. medio-bassa).';
    if (lMsg.includes('orphelins') && lMsg.includes('moderate low frequency'))
        return 'Orphelins sta uscendo poco (freq. medio-bassa).';
    if (lMsg.includes('sector') && lMsg.includes('moderate low frequency'))
        return 'Questo settore sta uscendo poco.';
    if (lMsg.includes('sector') && lMsg.includes('high frequency'))
        return 'Questo settore sta uscendo molto.';

    // Streak patterns
    if (lMsg.includes('consecutive') && lMsg.includes('first') && lMsg.includes('dozen')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} uscite di fila nella 1ª Dozzina (1-12).`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('second') && lMsg.includes('dozen')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} uscite di fila nella 2ª Dozzina (13-24).`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('third') && lMsg.includes('dozen')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} uscite di fila nella 3ª Dozzina (25-36).`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('low') && lMsg.includes('high_low')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} uscite di fila su BASSI (1-18).`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('high') && lMsg.includes('high_low')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} uscite di fila su ALTI (19-36).`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('red')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} Rossi di fila.`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('black')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} Neri di fila.`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('even')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} Pari di fila.`;
    }
    if (lMsg.includes('consecutive') && lMsg.includes('odd')) {
        const count = lMsg.match(/\d+/)?.[0] || 'Vari';
        return `${count} Dispari di fila.`;
    }

    // Bias
    if (lMsg.includes('bias detected')) return 'Anomalia statistica rilevata.';

    // Fallback cleanup
    return msg
        .replace(/first/gi, '1ª')
        .replace(/second/gi, '2ª')
        .replace(/third/gi, '3ª')
        .replace(/dozen/gi, 'Dozzina')
        .replace(/column/gi, 'Colonna')
        .replace(/high_low/gi, 'Alto/Basso')
        .replace(/sector/gi, 'settore')
        .replace(/showing/gi, 'mostra')
        .replace(/moderate/gi, 'moderata')
        .replace(/frequency/gi, 'frequenza')
        .replace(/low/gi, 'bassa')
        .replace(/high/gi, 'alta');
}

function getPatternAdvice(msg: string): string {
    const lMsg = msg.toLowerCase();
    if (lMsg.includes('red')) return 'Punta sul NERO.';
    if (lMsg.includes('black')) return 'Punta sul ROSSO.';
    if (lMsg.includes('even')) return 'Punta sui DISPARI.';
    if (lMsg.includes('odd')) return 'Punta sui PARI.';
    if (lMsg.includes('high') && !lMsg.includes('frequency')) return 'Punta sui BASSI (1-18).';
    if (lMsg.includes('low') && !lMsg.includes('frequency')) return 'Punta sugli ALTI (19-36).';
    if (lMsg.includes('first') && lMsg.includes('dozen')) return 'Punta su 2ª o 3ª Dozzina.';
    if (lMsg.includes('second') && lMsg.includes('dozen')) return 'Punta su 1ª o 3ª Dozzina.';
    if (lMsg.includes('third') && lMsg.includes('dozen')) return 'Punta su 1ª o 2ª Dozzina.';
    if (lMsg.includes('voisins')) return 'NON puntare sui numeri vicini allo zero (0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35).';
    if (lMsg.includes('tiers')) return 'NON puntare sui numeri opposti allo zero (5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36).';
    if (lMsg.includes('orphelins')) return 'NON puntare sugli orfanelli (1, 6, 9, 14, 17, 20, 31, 34).';

    return 'Attendi conferma o cambia strategia.';
}

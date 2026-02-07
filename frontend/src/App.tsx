import { useState, useEffect, useCallback } from 'react';
import {
  API_BASE,
  getSpins,
  getPredictions,
  getAdvancedPredictions,
  getPatternAnalysis,
  getStatisticalAnalysis,
  clearSpins,
  PredictionsResponse,
  AdvancedPredictionsResponse,
  PatternAnalysisResponse,
  StatisticalAnalysisResponse,
} from './api';
import { SpinInput } from './components/SpinInput';
import { SpinHistory } from './components/SpinHistory';
import { PredictionsPanel } from './components/PredictionsPanel';
import { AdvancedPredictions } from './components/AdvancedPredictions';
import { PatternAnalysis } from './components/PatternAnalysis';
import { StatisticsPanel } from './components/StatisticsPanel';

type Tab = 'basic' | 'advanced' | 'patterns' | 'statistics';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('basic');
  const [spins, setSpins] = useState<{ spins: Array<{ number: number; color: string; timestamp: string }>; total: number }>({ spins: [], total: 0 });
  const [clearing, setClearing] = useState(false);

  // Basic predictions
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null);
  const [loadingPreds, setLoadingPreds] = useState(true);
  const [errorPreds, setErrorPreds] = useState<string | null>(null);

  // Advanced predictions
  const [advancedPreds, setAdvancedPreds] = useState<AdvancedPredictionsResponse | null>(null);
  const [loadingAdvanced, setLoadingAdvanced] = useState(true);
  const [errorAdvanced, setErrorAdvanced] = useState<string | null>(null);

  // Pattern analysis
  const [patterns, setPatterns] = useState<PatternAnalysisResponse | null>(null);
  const [loadingPatterns, setLoadingPatterns] = useState(true);
  const [errorPatterns, setErrorPatterns] = useState<string | null>(null);

  // Statistical analysis
  const [statistics, setStatistics] = useState<StatisticalAnalysisResponse | null>(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const [errorStats, setErrorStats] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErrorPreds(null);
    setErrorAdvanced(null);
    setErrorPatterns(null);
    setErrorStats(null);

    try {
      const [spinsRes, predsRes] = await Promise.all([
        getSpins(100),
        getPredictions(),
      ]);
      setSpins(spinsRes);
      setPredictions(predsRes);
      setLoadingPreds(false);

      // Fetch advanced data
      try {
        const advRes = await getAdvancedPredictions();
        setAdvancedPreds(advRes);
      } catch {
        setErrorAdvanced('Impossibile caricare predizioni avanzate');
      }
      setLoadingAdvanced(false);

      try {
        const patRes = await getPatternAnalysis();
        setPatterns(patRes);
      } catch {
        setErrorPatterns('Impossibile caricare analisi pattern');
      }
      setLoadingPatterns(false);

      try {
        const statsRes = await getStatisticalAnalysis();
        setStatistics(statsRes);
      } catch {
        setErrorStats('Impossibile caricare analisi statistica');
      }
      setLoadingStats(false);

    } catch {
      setErrorPreds('Backend non raggiungibile. Su Render (piano free) il server può impiegare 30-60 secondi ad avviarsi. Attendi e clicca Riprova, oppure apri direttamente: ' + API_BASE);
      setLoadingPreds(false);
      setLoadingAdvanced(false);
      setLoadingPatterns(false);
      setLoadingStats(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onSpinAdded = useCallback(() => {
    setLoadingPreds(true);
    setLoadingAdvanced(true);
    setLoadingPatterns(true);
    setLoadingStats(true);
    refresh();
  }, [refresh]);

  const handleClear = useCallback(async () => {
    if (!confirm('Sei sicuro di voler cancellare tutta la cronologia? Questa azione non può essere annullata.')) {
      return;
    }

    setClearing(true);
    try {
      await clearSpins();
      // Reset all state
      setSpins({ spins: [], total: 0 });
      setPredictions(null);
      setAdvancedPreds(null);
      setPatterns(null);
      setStatistics(null);
      setLoadingPreds(true);
      setLoadingAdvanced(true);
      setLoadingPatterns(true);
      setLoadingStats(true);
      refresh();
    } catch (e) {
      alert('Errore durante la cancellazione');
    } finally {
      setClearing(false);
    }
  }, [refresh]);

  const lastSpin = spins.spins.length > 0
    ? { number: spins.spins[0].number, color: spins.spins[0].color }
    : null;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>
          <span className="logo">🎰</span>
          Roulette ML
          <span className="version">v2.0</span>
        </h1>
        <p className="subtitle">
          Sistema AI avanzato per analisi e predizioni roulette europea
        </p>
        {spins.total > 0 && (
          <button
            className="clear-btn"
            onClick={handleClear}
            disabled={clearing}
          >
            {clearing ? '⏳ Cancellando...' : '🗑️ Cancella Cronologia'}
          </button>
        )}
      </header>

      <SpinInput onSpinAdded={onSpinAdded} lastSpin={lastSpin} />

      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'advanced' ? 'active' : ''}`}
          onClick={() => setActiveTab('advanced')}
        >
          <span className="tab-icon">🧠</span>
          AI Ensemble
        </button>
        <button
          className={`tab-btn ${activeTab === 'patterns' ? 'active' : ''}`}
          onClick={() => setActiveTab('patterns')}
        >
          <span className="tab-icon">📈</span>
          Pattern
        </button>
        <button
          className={`tab-btn ${activeTab === 'statistics' ? 'active' : ''}`}
          onClick={() => setActiveTab('statistics')}
        >
          <span className="tab-icon">📊</span>
          Statistiche
        </button>
        <button
          className={`tab-btn ${activeTab === 'basic' ? 'active' : ''}`}
          onClick={() => setActiveTab('basic')}
        >
          <span className="tab-icon">📋</span>
          Base
        </button>
      </nav>

      {/* Tab Content */}
      <main className="tab-content">
        {activeTab === 'basic' && (
          <PredictionsPanel
            data={predictions}
            loading={loadingPreds}
            error={errorPreds}
            onRetry={refresh}
          />
        )}

        {activeTab === 'advanced' && (
          <AdvancedPredictions
            data={advancedPreds}
            loading={loadingAdvanced}
            error={errorAdvanced}
          />
        )}

        {activeTab === 'patterns' && (
          <PatternAnalysis
            data={patterns}
            loading={loadingPatterns}
            error={errorPatterns}
          />
        )}

        {activeTab === 'statistics' && (
          <StatisticsPanel
            data={statistics}
            loading={loadingStats}
            error={errorStats}
          />
        )}
      </main>

      <SpinHistory spins={spins.spins} total={spins.total} />

      <footer className="app-footer">
        <p>
          ⚠️ La roulette è un gioco d'azzardo casuale.
          Queste predizioni sono a scopo educativo e di intrattenimento.
        </p>
      </footer>
    </div>
  );
}

export default App;

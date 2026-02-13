import { useState, useEffect, useCallback } from 'react';
import {
  API_BASE,
  getSpins,
  getAdvancedPredictions,
  getPatternAnalysis,
  getStatisticalAnalysis,
  getWheelClustering,
  clearSpins,
  AdvancedPredictionsResponse,
  PatternAnalysisResponse,
  StatisticalAnalysisResponse,
  WheelClusteringResponse,
} from './api';
import { SpinInput } from './components/SpinInput';
import { SpinHistory } from './components/SpinHistory';
import { AdvancedPredictions } from './components/AdvancedPredictions';
import { PatternAnalysis } from './components/PatternAnalysis';
import { WheelAnalysis } from './components/WheelAnalysis';
import { ProbabilitySummary } from './components/ProbabilitySummary';
import { StatisticsPanel } from './components/StatisticsPanel';
import { AIStatisticsSection } from './components/AIStatisticsSection';
import { RouletteTableVisualization } from './components/RouletteTableVisualization';
import { ModelStatusHero } from './components/ModelStatusHero';

type Tab = 'riepilogo' | 'previsioni' | 'tavolo' | 'statistiche' | 'pattern' | 'storico';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('riepilogo');
  const [spins, setSpins] = useState<{ spins: Array<{ number: number; color: string; timestamp: string }>; total: number }>({ spins: [], total: 0 });
  const [clearing, setClearing] = useState(false);

  // Advanced predictions (ensemble)
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

  // Wheel clustering analysis
  const [wheelClustering, setWheelClustering] = useState<WheelClusteringResponse | null>(null);
  const [loadingWheel, setLoadingWheel] = useState(true);
  const [errorWheel, setErrorWheel] = useState<string | null>(null);
  const [errorStats, setErrorStats] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErrorAdvanced(null);
    setErrorPatterns(null);
    setErrorStats(null);
    setErrorWheel(null);

    try {
      // 1. Fetch spins first (fast & essential)
      const spinsRes = await getSpins(100);
      setSpins(spinsRes);
    } catch {
      setErrorAdvanced('Backend non raggiungibile. Su Render (piano free) il server può impiegare 30-60 secondi ad avviarsi. Attendi e clicca Riprova, oppure apri direttamente: ' + API_BASE);
      setLoadingAdvanced(false);
      setLoadingPatterns(false);
      setLoadingStats(false);
      setLoadingWheel(false);
      return;
    }

    // 2. Fire other requests in parallel (independent loading states)

    // Advanced Predictions (AI Ensemble - Slowest)
    getAdvancedPredictions()
      .then(res => setAdvancedPreds(res))
      .catch(() => setErrorAdvanced('Impossibile caricare predizioni avanzate'))
      .finally(() => setLoadingAdvanced(false));

    // Pattern Analysis
    getPatternAnalysis()
      .then(res => setPatterns(res))
      .catch(() => setErrorPatterns('Impossibile caricare analisi pattern'))
      .finally(() => setLoadingPatterns(false));

    // Statistical Analysis
    getStatisticalAnalysis()
      .then(res => setStatistics(res))
      .catch(() => setErrorStats('Impossibile caricare analisi statistica'))
      .finally(() => setLoadingStats(false));

    // Wheel Analysis
    getWheelClustering()
      .then(res => setWheelClustering(res))
      .catch(() => setErrorWheel('Impossibile caricare analisi ruota'))
      .finally(() => setLoadingWheel(false));

  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onSpinAdded = useCallback(() => {
    setLoadingAdvanced(true);
    setLoadingPatterns(true);
    setLoadingStats(true);
    setLoadingWheel(true);
    refresh();
  }, [refresh]);

  const handleClear = useCallback(async () => {
    if (!confirm('Sei sicuro di voler cancellare tutta la cronologia? Questa azione non può essere annullata.')) {
      return;
    }

    setClearing(true);
    try {
      await clearSpins();
      setSpins({ spins: [], total: 0 });
      setAdvancedPreds(null);
      setPatterns(null);
      setStatistics(null);
      setWheelClustering(null);
      setLoadingAdvanced(true);
      setLoadingPatterns(true);
      setLoadingWheel(true);
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

        <div style={{ margin: '1rem 0' }}>
          <ModelStatusHero
            info={advancedPreds?.model_info || null}
            loading={loadingAdvanced}
          />
        </div>

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

      {/* Tab Navigation: Previsioni | Statistiche */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'riepilogo' ? 'active' : ''}`}
          onClick={() => setActiveTab('riepilogo')}
        >
          <span className="tab-icon">💎</span>
          Riepilogo
        </button>
        <button
          className={`tab-btn ${activeTab === 'previsioni' ? 'active' : ''}`}
          onClick={() => setActiveTab('previsioni')}
        >
          <span className="tab-icon">🎯</span>
          Previsioni
        </button>
        <button
          className={`tab-btn ${activeTab === 'tavolo' ? 'active' : ''}`}
          onClick={() => setActiveTab('tavolo')}
        >
          <span className="tab-icon">🎲</span>
          Tavolo
        </button>
        <button
          className={`tab-btn ${activeTab === 'pattern' ? 'active' : ''}`}
          onClick={() => setActiveTab('pattern')}
        >
          <span className="tab-icon">🧩</span>
          Pattern
        </button>
        <button
          className={`tab-btn ${activeTab === 'statistiche' ? 'active' : ''}`}
          onClick={() => setActiveTab('statistiche')}
        >
          <span className="tab-icon">📊</span>
          Statistiche
        </button>
        <button
          className={`tab-btn ${activeTab === 'storico' ? 'active' : ''}`}
          onClick={() => setActiveTab('storico')}
        >
          <span className="tab-icon">📜</span>
          Storico
        </button>
      </nav>

      {/* Tab Content */}
      <main className="tab-content">
        {activeTab === 'riepilogo' && (
          <div className="page-riepilogo">
            <ProbabilitySummary
              data={advancedPreds}
              patternData={patterns}
              loading={loadingAdvanced}
            />
          </div>
        )}

        {activeTab === 'previsioni' && (
          <div className="page-predictions">
            <section className="predictions-block">
              <AdvancedPredictions
                data={advancedPreds}
                patternData={patterns}
                loading={loadingAdvanced}
                error={errorAdvanced}
                onRetry={refresh}
              />
            </section>
          </div>
        )}

        {activeTab === 'pattern' && (
          <div className="page-patterns">
            <section className="predictions-block">
              <PatternAnalysis
                data={patterns}
                loading={loadingPatterns}
                error={errorPatterns}
              />
            </section>
            <section className="predictions-block" style={{ marginTop: '2rem' }}>
              <WheelAnalysis
                data={wheelClustering}
                loading={loadingWheel}
                error={errorWheel}
              />
            </section>
          </div>
        )}

        {activeTab === 'statistiche' && (
          <div className="page-statistics">
            <section className="statistics-block">
              <AIStatisticsSection
                modelInfo={advancedPreds?.model_info ?? null}
                loading={loadingAdvanced}
                error={errorAdvanced}
              />
            </section>
            <section className="statistics-block">
              <StatisticsPanel
                data={statistics}
                loading={loadingStats}
                error={errorStats}
              />
            </section>
          </div>
        )}

        {activeTab === 'tavolo' && (
          <div className="page-table">
            <RouletteTableVisualization data={advancedPreds} />
          </div>
        )}

        {activeTab === 'storico' && (
          <div className="page-history">
            <SpinHistory spins={spins.spins} total={spins.total} onSpinDeleted={refresh} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          ⚠️ La roulette è un gioco d'azzardo casuale.
          Queste predizioni sono a scopo educativo e di intrattenimento.
        </p>
      </footer>
    </div >
  );
}

export default App;

import { ModelInfo } from '../api';

interface Props {
  modelInfo: ModelInfo | null;
  loading: boolean;
  error: string | null;
}

const modelColors: Record<string, string> = {
  DeepMLP: '#8b5cf6',
  RandomForest: '#10b981',
  GradientBoosting: '#f59e0b',
  XGBoost: '#ec4899',
};

export function AIStatisticsSection({ modelInfo, loading, error }: Props) {
  if (loading) {
    return (
      <div className="stat-card-glass loading-state" style={{ minHeight: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="spinner"></div>
        <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Caricamento informazioni modelli...</p>
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

  if (!modelInfo) {
    return (
      <div className="stat-card-glass info-state">
        <span className="info-icon" style={{ fontSize: '2rem' }}>🤖</span>
        <p>Nessuna informazione modelli disponibile</p>
      </div>
    );
  }

  const { models, total_samples, trained, min_samples_required, retrain_interval } = modelInfo;
  const entries = Object.entries(models);

  return (
    <div className="ai-statistics-section">
      <h2 className="section-title">
        <span className="icon">🤖</span>
        Statistiche sull'Ensemble AI
      </h2>

      <div className="info-box-premium">
        <h4 style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          💡 Come funziona l'IA
        </h4>
        <ul style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 1.6, opacity: 0.9 }}>
          <li><strong>Ensemble</strong> = Un "team" di cervelli AI. La previsione finale è il voto di maggioranza pesato.</li>
          <li><strong>Peso</strong> = Quanto ti "fidi" di quel modello specifico basato sulle sue performance passate.</li>
          <li><strong>Campioni</strong> = Più dati hai ({total_samples}), più l'AI diventa intelligente.</li>
        </ul>
      </div>

      <div className="stat-card-glass">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span className={`status-pill ${trained ? 'success' : 'warning'}`} style={{ fontSize: '1rem', padding: '0.5rem 1.2rem' }}>
              {trained ? '✓ ENSEMBLE ADDESTRATO' : '⚠ IN ATTESA DI DATI'}
            </span>
            {!trained && (
              <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                (Servono min. {min_samples_required} spin)
              </span>
            )}
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>Campioni Totali</div>
            <div className="stat-value-big" style={{ fontSize: '2rem' }}>{total_samples}</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.6, marginTop: '0.2rem' }}>
              Retrain automatico ogni {retrain_interval} nuovi dati
            </div>
          </div>
        </div>
      </div>

      <div className="models-stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
        {entries.map(([name, info]) => (
          <div
            key={name}
            className="stat-card-glass"
            style={{
              margin: 0,
              padding: '1.5rem',
              borderLeft: `4px solid ${modelColors[name] || '#6b7280'}`,
              background: 'rgba(255, 255, 255, 0.03)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'white' }}>{name}</h4>
              <div
                className="capsule-dot"
                style={{
                  backgroundColor: modelColors[name] || '#6b7280',
                  width: '12px', height: '12px',
                  boxShadow: `0 0 10px ${modelColors[name] || '#6b7280'}`
                }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Stato</span>
                <span className={info.trained ? 'status-pill success' : 'status-pill warning'} style={{ padding: '0.2rem 0.6rem', fontSize: '0.7rem' }}>
                  {info.available ? (info.trained ? 'Ready' : 'Not Trained') : 'Offline'}
                </span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Peso Ensemble</span>
                <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'white' }}>
                  {(info.weight * 100).toFixed(0)}%
                </span>
              </div>

              <div className="capsule-bar-track" style={{ height: '4px', background: 'rgba(255,255,255,0.1)' }}>
                <div
                  className="capsule-bar-fill"
                  style={{
                    width: `${info.weight * 100}%`,
                    backgroundColor: modelColors[name] || '#6b7280',
                    boxShadow: `0 0 8px ${modelColors[name] || '#6b7280'}`
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

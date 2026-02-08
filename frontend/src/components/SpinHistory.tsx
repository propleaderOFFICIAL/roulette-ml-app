import { useState } from 'react';
import { Spin, deleteSpin } from '../api';

interface SpinHistoryProps {
  spins: Spin[];
  total: number;
  onSpinDeleted?: () => void;
}

export function SpinHistory({ spins, total, onSpinDeleted }: SpinHistoryProps) {
  const [deletingTimestamp, setDeletingTimestamp] = useState<string | null>(null);

  const handleDelete = async (timestamp: string) => {
    if (deletingTimestamp) return;

    if (!confirm('Sei sicuro di voler eliminare questo spin?')) return;

    setDeletingTimestamp(timestamp);
    try {
      await deleteSpin(timestamp);
      if (onSpinDeleted) {
        onSpinDeleted();
      }
    } catch (error) {
      console.error('Error deleting spin:', error);
      alert('Errore durante l\'eliminazione dello spin');
    } finally {
      setDeletingTimestamp(null);
    }
  };

  return (
    <div className="stat-card-glass" style={{ padding: '0', overflow: 'hidden' }}>
      <div style={{ padding: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="section-title" style={{ margin: 0, fontSize: '1.3rem' }}>
          <span className="icon">📜</span>
          Storico Uscite
        </h2>
        <span style={{ background: 'rgba(255,255,255,0.1)', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.85rem' }}>
          {total} totali
        </span>
      </div>

      {spins.length === 0 ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <p>Nessuna uscita registrata.</p>
          <p style={{ fontSize: '0.9rem', opacity: 0.7 }}>Aggiungi un numero per iniziare a tracciare lo storico.</p>
        </div>
      ) : (
        <div className="table-responsive" style={{ maxHeight: '600px', overflowY: 'auto' }}>
          <table className="premium-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead style={{ position: 'sticky', top: 0, background: 'rgba(17, 24, 39, 0.95)', zIndex: 10, backdropFilter: 'blur(5px)' }}>
              <tr>
                <th style={{ padding: '1rem', textAlign: 'left', color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px' }}>#</th>
                <th style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Numero</th>
                <th style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Colore</th>
                <th style={{ padding: '1rem', textAlign: 'right', color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Ora</th>
                <th style={{ padding: '1rem', width: '50px' }}></th>
              </tr>
            </thead>
            <tbody>
              {spins.map((s, i) => {
                const isDeleting = deletingTimestamp === s.timestamp;
                const index = total - spins.length + i + 1;

                return (
                  <tr key={`${s.timestamp}-${i}`} style={{
                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                    transition: 'background 0.2s',
                    opacity: isDeleting ? 0.5 : 1
                  }}
                    className="history-row"
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '1rem', color: 'var(--text-muted)', fontFamily: 'Roboto Mono' }}>{index}</td>
                    <td style={{ padding: '1rem', textAlign: 'center' }}>
                      <span style={{
                        fontSize: '1.2rem',
                        fontWeight: 700,
                        color: 'white',
                        display: 'inline-block',
                        width: '32px',
                        height: '32px',
                        lineHeight: '32px',
                        borderRadius: '50%',
                        background: s.number === 0 ? '#22c55e' : (['1', '3', '5', '7', '9', '12', '14', '16', '18', '19', '21', '23', '25', '27', '30', '32', '34', '36'].includes(String(s.number)) ? '#ef4444' : '#4b5563'),
                        boxShadow: '0 2px 5px rgba(0,0,0,0.3)'
                      }}>
                        {s.number}
                      </span>
                    </td>
                    <td style={{ padding: '1rem', textAlign: 'center' }}>
                      <span className={`predicted-color-chip`} style={{
                        fontSize: '0.75rem',
                        padding: '0.3rem 0.8rem',
                        backgroundColor: s.color === 'red' ? '#ef4444' : (s.color === 'black' ? '#4b5563' : '#22c55e'),
                        boxShadow: `0 0 10px ${s.color === 'red' ? '#ef4444' : (s.color === 'black' ? 'rgba(255,255,255,0.2)' : '#22c55e')}`
                      }}>
                        {s.color.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '1rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.9rem' }}>
                      {new Date(s.timestamp).toLocaleTimeString('it-IT')}
                    </td>
                    <td style={{ padding: '1rem', textAlign: 'right' }}>
                      <button
                        onClick={() => handleDelete(s.timestamp)}
                        disabled={isDeleting}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#ef4444',
                          cursor: 'pointer',
                          padding: '5px',
                          opacity: 0.7,
                          transition: 'opacity 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                        onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                        title="Elimina spin"
                      >
                        {isDeleting ? '...' : (
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z" />
                            <path fillRule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z" />
                          </svg>
                        )}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

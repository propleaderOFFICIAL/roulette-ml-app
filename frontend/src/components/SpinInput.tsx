import { useState, FormEvent } from 'react';
import { addSpin } from '../api';

interface SpinInputProps {
  onSpinAdded: () => void;
  lastSpin: { number: number; color: string } | null;
}

const colorMap: Record<string, string> = {
  red: '#ef4444',
  black: '#1f2937',
  green: '#22c55e',
};

export function SpinInput({ onSpinAdded, lastSpin }: SpinInputProps) {
  const [number, setNumber] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const n = parseInt(number, 10);
    if (isNaN(n) || n < 0 || n > 36) {
      setError('Inserisci un numero tra 0 e 36');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await addSpin(n);
      setNumber('');
      onSpinAdded();
    } catch {
      setError('Errore di connessione al server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="spin-input-container">
      <h2>🎰 Registra Uscita</h2>
      <p className="input-hint">Inserisci il numero che è appena uscito alla roulette</p>

      <form onSubmit={handleSubmit} className="input-group">
        <input
          type="number"
          min={0}
          max={36}
          value={number}
          onChange={(e) => setNumber(e.target.value)}
          disabled={loading}
          placeholder="0-36"
          autoFocus
        />
        <button type="submit" disabled={loading}>
          {loading ? '...' : '➕ Aggiungi'}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}

      {lastSpin && (
        <div className="last-spin">
          <span>Ultima uscita:</span>
          <div
            className="last-spin-number"
            style={{ backgroundColor: colorMap[lastSpin.color] }}
          >
            {lastSpin.number}
          </div>
          <span className="last-spin-color">{lastSpin.color}</span>
        </div>
      )}
    </div>
  );
}

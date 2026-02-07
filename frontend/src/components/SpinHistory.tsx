import { Spin } from '../api';

interface SpinHistoryProps {
  spins: Spin[];
  total: number;
}

export function SpinHistory({ spins, total }: SpinHistoryProps) {
  return (
    <div className="card">
      <h2>Storico uscite {total > 0 && `(${total} totali)`}</h2>
      {spins.length === 0 ? (
        <p style={{ color: '#94a3b8' }}>Nessuna uscita registrata. Aggiungi un numero per iniziare.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Numero</th>
                <th>Colore</th>
                <th>Ora</th>
              </tr>
            </thead>
            <tbody>
              {spins.map((s, i) => (
                <tr key={`${s.timestamp}-${i}`}>
                  <td>{total - spins.length + i + 1}</td>
                  <td><strong>{s.number}</strong></td>
                  <td><span className={`label-${s.color}`}>{s.color}</span></td>
                  <td style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                    {new Date(s.timestamp).toLocaleTimeString('it-IT')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

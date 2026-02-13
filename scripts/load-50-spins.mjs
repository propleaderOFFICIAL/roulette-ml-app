#!/usr/bin/env node
/**
 * Carica gli ultimi numeri (roulette 0-36) nell'API.
 * Ordine: dal più vecchio al più recente.
 * Uso: node scripts/load-50-spins.mjs [URL_API]
 */

const API_BASE = (process.env.API_URL || process.argv[2] || 'http://localhost:8000').replace(/\/$/, '');

// Last Results: 50 numeri, 5 righe x 10. Ordine dal più vecchio al più recente (9 = ultimo).
const NUMBERS = [
  17, 31, 25, 17, 32, 32, 24, 30, 23, 32,
  2, 28, 34, 16, 13, 0, 19, 30, 29, 2,
  15, 30, 18, 11, 2, 26, 1, 9, 13, 7,
  32, 17, 33, 31, 1, 23, 12, 28, 21, 19,
  14, 0, 28, 23, 11, 32, 8, 28, 0, 9,
];

async function main() {
  console.log(`Caricamento di ${NUMBERS.length} uscite su ${API_BASE}...`);
  let ok = 0;
  let err = 0;
  for (let i = 0; i < NUMBERS.length; i++) {
    const n = NUMBERS[i];
    try {
      const res = await fetch(`${API_BASE}/spins`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ number: n }),
      });
      if (!res.ok) throw new Error(await res.text());
      ok++;
      process.stdout.write(`  ${i + 1}/${NUMBERS.length} → ${n} ${res.ok ? '✓' : ''}\n`);
    } catch (e) {
      err++;
      console.error(`  ${i + 1}/${NUMBERS.length} → ${n} ERRORE:`, e.message);
    }
    // piccola pausa per non stressare il server
    await new Promise((r) => setTimeout(r, 80));
  }
  console.log(`\nFatto: ${ok} ok, ${err} errori.`);
  if (err > 0) process.exit(1);
}

main();

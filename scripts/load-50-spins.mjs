#!/usr/bin/env node
/**
 * Carica 110 numeri casuali (roulette 0-36) nell'API per test completi.
 * Seed fisso 42 = stessa sequenza ogni volta.
 * Uso: node scripts/load-50-spins.mjs [URL_API]
 */

const API_BASE = (process.env.API_URL || process.argv[2] || 'http://localhost:8000').replace(/\/$/, '');

// 110 numeri casuali (seed 42), ordine dal più vecchio al più recente
const NUMBERS = [
  7, 1, 17, 15, 14, 8, 6, 34, 5, 27, 2, 1, 5, 13, 14, 32, 1, 35, 12, 34, 26, 14, 28, 17, 0, 10, 27, 21, 17, 9, 13, 21, 6, 5, 24, 6, 22, 22, 16, 2, 29, 34, 7, 24, 5, 35, 18, 23, 36, 12, 4, 2, 14, 18, 5, 14, 6, 24, 17, 29, 23, 10, 23, 22, 13, 17, 4, 10, 34, 15, 10, 29, 24, 17, 35, 14, 20, 3, 14, 2, 20, 25, 17, 4, 13, 36, 20, 13, 31, 25, 29, 9, 16, 8, 15, 35, 34, 16, 27, 25, 23, 14, 8, 32, 31, 5, 3, 7, 9, 10,
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

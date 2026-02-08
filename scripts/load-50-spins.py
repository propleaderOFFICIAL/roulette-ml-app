#!/usr/bin/env python3
"""
Carica 110 numeri casuali (roulette 0-36) nell'API per test completi.
Seed fisso 42 = stessa sequenza ogni volta.
Uso: python scripts/load-50-spins.py [URL_API]
"""
import sys
import time
import urllib.request
import json

API_BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

# 110 numeri casuali (seed 42), ordine dal più vecchio al più recente
NUMBERS = [
    7, 1, 17, 15, 14, 8, 6, 34, 5, 27, 2, 1, 5, 13, 14, 32, 1, 35, 12, 34, 26, 14, 28, 17, 0, 10, 27, 21, 17, 9, 13, 21, 6, 5, 24, 6, 22, 22, 16, 2, 29, 34, 7, 24, 5, 35, 18, 23, 36, 12, 4, 2, 14, 18, 5, 14, 6, 24, 17, 29, 23, 10, 23, 22, 13, 17, 4, 10, 34, 15, 10, 29, 24, 17, 35, 14, 20, 3, 14, 2, 20, 25, 17, 4, 13, 36, 20, 13, 31, 25, 29, 9, 16, 8, 15, 35, 34, 16, 27, 25, 23, 14, 8, 32, 31, 5, 3, 7, 9, 10,
]


def main():
    print(f"Caricamento di {len(NUMBERS)} uscite su {API_BASE}...")
    ok = err = 0
    for i, n in enumerate(NUMBERS):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/spins",
                data=json.dumps({"number": n}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req) as res:
                if res.status == 200:
                    ok += 1
                    print(f"  {i + 1}/{len(NUMBERS)} → {n} ✓")
                else:
                    raise Exception(f"HTTP {res.status}")
        except Exception as e:
            err += 1
            print(f"  {i + 1}/{len(NUMBERS)} → {n} ERRORE: {e}")
        time.sleep(0.08)
    print(f"\nFatto: {ok} ok, {err} errori.")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()

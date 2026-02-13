#!/usr/bin/env python3
"""
Carica i numeri estratti dallo screenshot nella roulette-ml-app locale.
Ordine: dal più vecchio al più recente (griglia 21 righe x 9 colonne).
Uso: python scripts/load-screenshot-spins.py [URL_API]
"""
import sys
import time
import urllib.request
import json

API_BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

# Numeri estratti dallo screenshot - ordine top-left → bottom-right (dal più vecchio al più recente)
NUMBERS = [
    20, 12, 36, 8, 32, 5, 4, 7, 26, 28, 27, 3, 19, 23, 8, 23, 1, 21,
    31, 26, 8, 10, 23, 15, 12, 5, 31, 20, 23, 31, 10, 26, 3, 0, 18, 27,
    5, 16, 3, 13, 0, 6, 20, 31, 35, 28, 28, 24, 6, 21, 22, 32, 0, 11,
    31, 24, 28, 5, 9, 16, 0, 7, 23, 31, 21, 27, 32, 18, 4, 33, 10, 30,
    32, 9, 4, 0, 10, 29, 36, 34, 20, 9, 4, 7, 28, 6, 14, 18, 16, 1,
    4, 36, 8, 0, 27, 9, 33, 19, 1, 13, 23, 15, 29, 0, 28, 5, 11, 13,
    3, 21, 10, 13, 33, 13, 7, 32, 13, 7, 3, 29, 11, 24, 23, 29, 14, 22,
    21, 0, 21, 28, 29, 11, 28, 3, 27, 8, 34, 32, 1, 29, 2, 0, 2, 15,
    21, 1, 2, 11, 29, 4, 13, 28, 35, 31, 22, 18, 0, 32, 19, 34, 10, 33,
    6, 12, 19, 22, 29, 15, 26, 8, 14, 11, 33, 3, 4, 36, 15, 6, 15, 16,
    28, 4, 24, 26, 11, 10, 3, 30, 4,
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
        time.sleep(0.05)
    print(f"\nFatto: {ok} ok, {err} errori.")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()

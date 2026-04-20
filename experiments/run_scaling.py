"""
run_scaling.py
──────────────
Experimento de strong scaling:
  ─ Corre la versión secuencial (baseline) y la paralela con 1, 2, 4, 8 cores.
  ─ Registra tiempos de pared y calcula speed-up.
  ─ Guarda resultados en data/scaling_results.csv.
  ─ Genera gráfica data/speedup.png.

Uso:
    python experiments/run_scaling.py [--size 1000] [--days 365] [--repeats 1]
"""

import argparse
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sequential.sir_sequential import run as run_seq
from parallel.sir_parallel     import run as run_par

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Experimento ───────────────────────────────────────────────────────────────

def run_experiment(size: int = 1000,
                   days: int = 365,
                   repeats: int = 1,
                   core_counts: list = None) -> pd.DataFrame:
    """
    Ejecuta el experimento de strong scaling.

    Parámetros
    ----------
    size        : tamaño de la grilla (side)
    days        : días a simular
    repeats     : repeticiones por configuración (se promedia)
    core_counts : lista de cores a probar; por defecto [1, 2, 4, 8] ∩ cpu_count

    Retorna
    -------
    pd.DataFrame con columnas: cores, mode, wall_time, speedup, efficiency
    """
    max_cores = mp.cpu_count()
    if core_counts is None:
        core_counts = [c for c in [1, 2, 4, 8] if c <= max_cores]
        if max_cores not in core_counts:
            core_counts.append(max_cores)
    core_counts = sorted(set(core_counts))

    records = []

    # ── Baseline secuencial ───────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Baseline SECUENCIAL  ({size}×{size}, {days} días)")
    print(f"{'─'*55}")
    t_seq_list = []
    for rep in range(repeats):
        r = run_seq(size=size, days=days, verbose=False, seed=42)
        t_seq_list.append(r['wall_time'])
        print(f"  rep {rep+1}/{repeats}: {r['wall_time']:.2f} s")
    t_seq = float(np.mean(t_seq_list))
    print(f"  ► media: {t_seq:.2f} s\n")
    records.append(dict(cores=1, mode="sequential",
                        wall_time=round(t_seq, 3),
                        speedup=1.0, efficiency=1.0))

    # ── Runs paralelos ────────────────────────────────────────────────────────
    for n in core_counts:
        print(f"{'─'*55}")
        print(f"  PARALELO  {n} workers  ({size}×{size}, {days} días)")
        print(f"{'─'*55}")
        t_par_list = []
        for rep in range(repeats):
            r = run_par(size=size, days=days, n_workers=n,
                        verbose=False, seed=42)
            t_par_list.append(r['wall_time'])
            print(f"  rep {rep+1}/{repeats}: {r['wall_time']:.2f} s")
        t_par    = float(np.mean(t_par_list))
        speedup  = t_seq / t_par
        eff      = speedup / n
        print(f"  ► media: {t_par:.2f} s  |  speed-up: {speedup:.2f}×  "
              f"|  eficiencia: {eff*100:.1f}%\n")
        records.append(dict(cores=n, mode="parallel",
                            wall_time=round(t_par, 3),
                            speedup=round(speedup, 4),
                            efficiency=round(eff, 4)))

    df = pd.DataFrame(records)

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    csv_path = DATA_DIR / "scaling_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV guardado → {csv_path}")

    # ── Graficar ──────────────────────────────────────────────────────────────
    _plot(df, size, days)
    return df


# ── Visualización ─────────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, size: int, days: int):
    cores   = df['cores'].values
    speedup = df['speedup'].values
    eff     = df['efficiency'].values * 100
    times   = df['wall_time'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Strong Scaling — SIR {size:,}×{size:,}, {days} días\n"
        f"({mp.cpu_count()} cores físicos disponibles)",
        fontsize=13, fontweight='bold'
    )

    palette = "#2563EB"

    # ── Panel 1: Speed-up ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(cores, speedup, "o-", color=palette, lw=2.5, ms=8,
            label="Medido", zorder=3)
    ax.plot(cores, cores,   "--", color="gray", alpha=0.5, lw=1.5,
            label="Ideal (lineal)")
    ax.fill_between(cores, speedup, cores,
                    where=(np.array(speedup) < np.array(cores, dtype=float)),
                    alpha=0.08, color=palette)
    for x, y in zip(cores, speedup):
        ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel("Número de cores")
    ax.set_ylabel("Speed-up")
    ax.set_title("Speed-up vs Cores")
    ax.set_xticks(cores)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    # ── Panel 2: Eficiencia paralela ──────────────────────────────────────────
    ax = axes[1]
    ax.bar([str(c) for c in cores], eff, color=palette, alpha=0.85,
           edgecolor='white', linewidth=0.8)
    ax.axhline(100, linestyle="--", color="gray", alpha=0.5, lw=1.5,
               label="100 % (ideal)")
    for x, y in enumerate(eff):
        ax.text(x, y + 1.5, f"{y:.1f}%", ha='center', fontsize=9)
    ax.set_xlabel("Número de cores")
    ax.set_ylabel("Eficiencia (%)")
    ax.set_title("Eficiencia Paralela")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)

    # ── Panel 3: Tiempo de pared ──────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar([str(c) for c in cores], times, color=palette, alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f} s", ha='center', fontsize=9)
    ax.set_xlabel("Número de cores")
    ax.set_ylabel("Tiempo de pared (s)")
    ax.set_title("Tiempo Total vs Cores")
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    out = DATA_DIR / "speedup.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Gráfica guardada → {out}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strong scaling – SIR 2D")
    parser.add_argument("--size",    type=int, default=1000)
    parser.add_argument("--days",    type=int, default=365)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cores",   nargs="+", type=int, default=None,
                        help="Lista de cores a probar (ej: 1 2 4 8)")
    args = parser.parse_args()

    df = run_experiment(size=args.size, days=args.days,
                        repeats=args.repeats, core_counts=args.cores)
    print("\n=== Tabla de resultados ===")
    print(df.to_string(index=False))

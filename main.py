"""
main.py
───────
Script maestro: ejecuta el pipeline completo del proyecto SIR 2D.

    python main.py [--quick] [--workers N] [--skip-animation]

Flags
─────
  --quick          Grilla 200×200, 60 días (prueba rápida ~10 s)
  --workers N      Máximo de cores para la versión paralela (default: cpu_count)
  --skip-animation No genera el GIF (ahorra ~1 min en grillas grandes)
  --skip-scaling   Omite experimentos de scaling (útil para debuggear)
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sequential.sir_sequential import validate_small, run as run_seq
from parallel.sir_parallel     import run as run_par
from experiments.run_scaling   import run_experiment
from visualization.animate     import (save_snapshots, load_snapshots,
                                       animate_side_by_side,
                                       plot_epidemic_curves)

DATA_DIR      = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def print_banner(text: str):
    bar = "─" * 58
    print(f"\n{bar}\n  {text}\n{bar}")


def main():
    parser = argparse.ArgumentParser(description="SIR 2D – pipeline completo")
    parser.add_argument("--quick",          action="store_true")
    parser.add_argument("--workers",        type=int, default=mp.cpu_count())
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--skip-scaling",   action="store_true")
    args = parser.parse_args()

    # ── Parámetros de simulación ──────────────────────────────────────────────
    if args.quick:
        SIZE = 200; DAYS = 60; SNAP_INT = 5
        print("  ► Modo RÁPIDO: 200×200, 60 días")
    else:
        SIZE = 1000; DAYS = 365; SNAP_INT = 7
        print("  ► Modo COMPLETO: 1 000×1 000, 365 días")

    N_WORKERS = min(args.workers, mp.cpu_count())

    print(f"  ► {N_WORKERS} workers paralelos / {mp.cpu_count()} disponibles")

    # ────────────────────────────────────────────────────────────────────────
    # 1. VALIDACIÓN
    # ────────────────────────────────────────────────────────────────────────
    print_banner("1 / 5  Validación (50×50, 30 días)")
    ok = validate_small(size=50, days=30)
    if not ok:
        print("  ⚠ Validación falló – verifica los parámetros del modelo.")

    # ────────────────────────────────────────────────────────────────────────
    # 2. SIMULACIÓN SECUENCIAL
    # ────────────────────────────────────────────────────────────────────────
    print_banner(f"2 / 5  Simulación SECUENCIAL  ({SIZE}×{SIZE}, {DAYS} días)")
    res_seq = run_seq(size=SIZE, days=DAYS,
                      snapshot_interval=SNAP_INT, verbose=True)
    save_snapshots(res_seq, tag="sequential")

    df_seq = res_seq['stats']
    print(f"\n  Pico de infectados : {df_seq['I'].max():>8,}  "
          f"(día {df_seq['I'].idxmax()})")
    print(f"  Infectados totales : {df_seq['cumulative_I'].iloc[-1]:>8,}")
    print(f"  Muertos totales    : {df_seq['D'].iloc[-1]:>8,}")
    print(f"  Tiempo de pared    : {res_seq['wall_time']:.2f} s")

    # ────────────────────────────────────────────────────────────────────────
    # 3. SIMULACIÓN PARALELA
    # ────────────────────────────────────────────────────────────────────────
    print_banner(f"3 / 5  Simulación PARALELA  "
                 f"({SIZE}×{SIZE}, {DAYS} días, {N_WORKERS} workers)")
    res_par = run_par(size=SIZE, days=DAYS, n_workers=N_WORKERS,
                      snapshot_interval=SNAP_INT, verbose=True)
    save_snapshots(res_par, tag="parallel")

    df_par = res_par['stats']
    print(f"\n  Pico de infectados : {df_par['I'].max():>8,}  "
          f"(día {df_par['I'].idxmax()})")
    print(f"  Infectados totales : {df_par['cumulative_I'].iloc[-1]:>8,}")
    print(f"  Muertos totales    : {df_par['D'].iloc[-1]:>8,}")
    print(f"  Tiempo de pared    : {res_par['wall_time']:.2f} s")

    # Guardar curvas epidémicas
    plot_epidemic_curves(df_seq, df_par)

    # Guardar CSVs de estadísticas
    df_seq.to_csv(DATA_DIR / "stats_sequential.csv", index=False)
    df_par.to_csv(DATA_DIR / "stats_parallel.csv",   index=False)
    print("\n  CSVs de estadísticas guardados en data/")

    # ────────────────────────────────────────────────────────────────────────
    # 4. EXPERIMENTOS DE SCALING
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_scaling:
        print_banner(f"4 / 5  Strong Scaling  ({SIZE}×{SIZE}, {DAYS} días)")
        df_scale = run_experiment(size=SIZE, days=DAYS, repeats=1)
        print("\n  Resultados de scaling:")
        print(df_scale.to_string(index=False))
    else:
        print_banner("4 / 5  Scaling – OMITIDO (--skip-scaling)")

    # ────────────────────────────────────────────────────────────────────────
    # 5. ANIMACIÓN
    # ────────────────────────────────────────────────────────────────────────
    if not args.skip_animation:
        print_banner("5 / 5  Generando animación side-by-side")
        snaps_seq = load_snapshots("sequential")
        snaps_par = load_snapshots("parallel")
        gif = animate_side_by_side(snaps_seq, snaps_par,
                                   interval_days=SNAP_INT,
                                   out_path=ROOT / "animation",
                                   fps=8)
        if gif:
            print(f"\n  ✓ Animación → {gif}")
    else:
        print_banner("5 / 5  Animación – OMITIDA (--skip-animation)")

    # ────────────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ────────────────────────────────────────────────────────────────────────
    t_seq = res_seq['wall_time']
    t_par = res_par['wall_time']
    speedup = t_seq / t_par

    print(f"\n{'═'*58}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*58}")
    print(f"  Tiempo secuencial  : {t_seq:>7.2f} s")
    print(f"  Tiempo paralelo    : {t_par:>7.2f} s  ({N_WORKERS} cores)")
    print(f"  Speed-up           : {speedup:>7.2f}×")
    print(f"  Eficiencia         : {speedup/N_WORKERS*100:>6.1f} %")
    print(f"\n  Archivos generados:")
    print(f"    data/stats_sequential.csv")
    print(f"    data/stats_parallel.csv")
    if not args.skip_scaling:
        print(f"    data/scaling_results.csv")
        print(f"    data/speedup.png")
    print(f"    data/epidemic_curves.png")
    if not args.skip_animation:
        print(f"    animation.gif  (y .mp4 si ffmpeg está disponible)")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    mp.freeze_support()   # necesario en Windows
    main()

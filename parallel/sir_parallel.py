"""
sir_parallel.py
───────────────
Versión paralela del modelo SIR 2-D usando multiprocessing.

Estrategia de paralelización
─────────────────────────────
1. La grilla (size × size) se divide en N franjas horizontales iguales.
2. Cada proceso worker recibe:
      ─ su franja real
      ─ ghost cells: la última fila de la franja anterior y la primera de la siguiente
3. El worker calcula la convolución sobre la franja + ghost cells
   (con mode='constant', cval=0) y devuelve sólo las filas reales actualizadas.
4. El proceso principal re-ensambla la grilla con np.concatenate.
5. Las estadísticas globales (S, I, R, D) se calculan sobre la grilla ensamblada
   (reducción paralela implícita mediante sumas de numpy).

Interfaz idéntica a sir_sequential.run() para facilitar comparaciones.
"""

import time
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sequential.sir_sequential import (
    initialize_grid, KERNEL,
    DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_MU,
    S, I, R, D
)


# ── Worker function (debe estar al nivel de módulo para pickle) ───────────────

def _step_strip(args: tuple) -> np.ndarray:
    """
    Actualiza una franja horizontal de la grilla.

    Argumentos (empaquetados en tupla para pool.map)
    ─────────────────────────────────────────────────
    strip_ghosts  : np.ndarray   franja + ghost rows (arriba y/o abajo)
    real_rows     : int          número de filas reales (sin ghost)
    has_top_ghost : bool         si hay fila ghost en la parte superior
    beta, gamma, mu : float      parámetros epidemiológicos
    rng_seed      : int          semilla para el generador aleatorio
    """
    (strip_ghosts, real_rows, has_top_ghost,
     beta, gamma, mu, rng_seed) = args

    rng = np.random.default_rng(rng_seed)

    # ── Contar vecinos infectados sobre la franja completa (con ghost cells) ──
    inf_mask = (strip_ghosts == I).astype(np.float32)
    nb_full  = convolve(inf_mask, KERNEL, mode="constant", cval=0)

    # ── Extraer sólo las filas reales ─────────────────────────────────────────
    r0     = 1 if has_top_ghost else 0
    r1     = r0 + real_rows
    nb_inf = nb_full[r0:r1]
    strip  = strip_ghosts[r0:r1]

    new_strip = strip.copy()

    # S → I
    susc  = (strip == S)
    p_inf = 1.0 - (1.0 - beta) ** nb_inf
    new_strip[susc & (rng.random(strip.shape) < p_inf)] = I

    # I → R
    inf     = (strip == I)
    recover = inf & (rng.random(strip.shape) < gamma)
    new_strip[recover] = R

    # I → D
    still_inf = inf & ~recover
    new_strip[still_inf & (rng.random(strip.shape) < mu)] = D

    return new_strip


# ── Utilidades de partición ───────────────────────────────────────────────────

def _make_slices(H: int, n: int) -> list:
    """
    Devuelve lista de (start_row, end_row) para n franjas horizontales.
    Las filas sobrantes (H % n) se reparten una a una en las primeras franjas.
    """
    base, rem = divmod(H, n)
    slices, s = [], 0
    for k in range(n):
        e = s + base + (1 if k < rem else 0)
        slices.append((s, e))
        s = e
    return slices


def _build_worker_args(grid: np.ndarray, slices: list,
                       beta: float, gamma: float, mu: float,
                       day: int) -> list:
    """Construye los argumentos para cada worker, añadiendo ghost cells."""
    H    = grid.shape[0]
    args = []
    for k, (s, e) in enumerate(slices):
        top    = (s > 0)
        bottom = (e < H)
        parts  = []
        if top:
            parts.append(grid[s - 1: s])   # ghost row superior
        parts.append(grid[s:e])             # franja real
        if bottom:
            parts.append(grid[e: e + 1])   # ghost row inferior
        strip = np.concatenate(parts, axis=0)
        # Semilla única por (día, worker) para reproducibilidad entre ejecuciones
        seed = (day * 9973 + k * 1000003) & 0xFFFF_FFFF
        args.append((strip, e - s, top, beta, gamma, mu, seed))
    return args


# ── Reducción de estadísticas globales ───────────────────────────────────────

def _global_stats(grid: np.ndarray) -> dict:
    """
    Calcula S, I, R, D sobre toda la grilla ensamblada.
    Esto constituye la reducción paralela: cada subarray ya fue procesado
    por un worker, y aquí simplemente sumamos sobre el array completo.
    """
    return dict(
        S=int((grid == S).sum()),
        I=int((grid == I).sum()),
        R=int((grid == R).sum()),
        D=int((grid == D).sum()),
    )


# ── Simulación paralela ───────────────────────────────────────────────────────

def run(size: int = 1000,
        days: int = 365,
        n_workers: int = 4,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        mu: float = DEFAULT_MU,
        seed: int = 42,
        snapshot_interval: int = 0,
        verbose: bool = True) -> dict:
    """
    Ejecuta la simulación SIR paralela.
    Misma interfaz que sequential.sir_sequential.run().

    Retorna
    -------
    dict con claves:
        'stats'      : pd.DataFrame  (day, S, I, R, D, cumulative_I)
        'snapshots'  : list[np.ndarray]
        'wall_time'  : float  (segundos)
        'final_grid' : np.ndarray
    """
    grid   = initialize_grid(size, seed)
    slices = _make_slices(size, n_workers)

    stats     = []
    snapshots = []
    cum_inf   = int((grid == I).sum())

    t0 = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        for day in range(days):
            # ── Estadísticas del día actual (reducción) ───────────────────────
            g = _global_stats(grid)
            stats.append(dict(day=day, cumulative_I=cum_inf, **g))

            if snapshot_interval > 0 and day % snapshot_interval == 0:
                snapshots.append(grid.copy())

            if verbose and day % 30 == 0:
                print(f"  Día {day:3d}: S={g['S']:>8,}  I={g['I']:>7,}  "
                      f"R={g['R']:>8,}  D={g['D']:>7,}")

            # ── Paso paralelo: split → map → reduce ───────────────────────────
            worker_args = _build_worker_args(grid, slices, beta, gamma, mu, day)
            strips      = pool.map(_step_strip, worker_args)
            grid        = np.concatenate(strips, axis=0)   # re-ensamblar

            new_i    = int((grid == I).sum())
            cum_inf += max(0, new_i - g['I'])

    wall_time = time.perf_counter() - t0

    if verbose:
        print(f"\n  ✓ Paralelo ({n_workers} workers) terminado en {wall_time:.2f} s")

    return dict(stats=pd.DataFrame(stats),
                snapshots=snapshots,
                wall_time=wall_time,
                final_grid=grid)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    workers = int(os.environ.get("SIR_WORKERS", mp.cpu_count()))
    print(f"=== Simulación paralela ({workers} workers, 1 000×1 000, 365 días) ===")
    result = run(n_workers=workers, snapshot_interval=7, verbose=True)
    df = result['stats']
    print(f"\n  Pico de infectados : {df['I'].max():>8,}  (día {df['I'].idxmax()})")
    print(f"  Infectados totales : {df['cumulative_I'].iloc[-1]:>8,}")
    print(f"  Muertos totales    : {df['D'].iloc[-1]:>8,}")
    print(f"  Tiempo de pared    : {result['wall_time']:.2f} s")

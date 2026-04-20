"""
sir_sequential.py
─────────────────
Modelo SIR estocástico en grilla 2-D (toroidal).

Estados:  S=0 (Susceptible)  I=1 (Infectado)
          R=2 (Recuperado)   D=3 (Muerto)

Reglas de transición (por celda, por día):
  S → I : P = 1 − (1 − β)^k   donde k = vecinos infectados (Moore, 8)
  I → R : P = γ
  I → D : P = μ  (sólo si no se recuperó en el mismo paso)
"""

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

# ── Constantes ────────────────────────────────────────────────────────────────
S, I, R, D = 0, 1, 2, 3

KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.float32)   # vecindad de Moore

# Parámetros epidemiológicos por defecto
DEFAULT_BETA  = 0.05    # probabilidad de contagio por vecino infectado
DEFAULT_GAMMA = 0.10    # probabilidad diaria de recuperación
DEFAULT_MU    = 0.005   # probabilidad diaria de muerte
# R0 teórico ≈ 8·β/γ = 8·0.05/0.10 = 4  (similar a un brote moderado)


# ── Inicialización ────────────────────────────────────────────────────────────

def initialize_grid(size: int, seed: int = 42) -> np.ndarray:
    """
    Crea una grilla size×size con un clúster infectado en el centro.
    radio del clúster ≈ 1 % del tamaño de la grilla.
    """
    grid = np.zeros((size, size), dtype=np.int8)
    r = max(2, size // 100)
    c = size // 2
    grid[c - r: c + r, c - r: c + r] = I
    return grid


# ── Núcleo de la simulación ───────────────────────────────────────────────────

def _count_infected_neighbors(grid: np.ndarray) -> np.ndarray:
    """Cuenta vecinos infectados para cada celda (convolución vectorizada)."""
    inf_mask = (grid == I).astype(np.float32)
    return convolve(inf_mask, KERNEL, mode="wrap")   # grilla toroidal


def step(grid: np.ndarray,
         beta: float, gamma: float, mu: float,
         rng: np.random.Generator) -> np.ndarray:
    """Avanza la grilla un día. Devuelve una NUEVA grilla (grid no se modifica)."""
    nb_inf   = _count_infected_neighbors(grid)
    new_grid = grid.copy()

    # ── S → I ────────────────────────────────────────────────────────────────
    susc  = (grid == S)
    p_inf = 1.0 - (1.0 - beta) ** nb_inf
    new_grid[susc & (rng.random(grid.shape) < p_inf)] = I

    # ── I → R ────────────────────────────────────────────────────────────────
    inf     = (grid == I)
    recover = inf & (rng.random(grid.shape) < gamma)
    new_grid[recover] = R

    # ── I → D (sólo los que NO se recuperaron) ────────────────────────────────
    still_inf = inf & ~recover
    new_grid[still_inf & (rng.random(grid.shape) < mu)] = D

    return new_grid


# ── Simulación completa ───────────────────────────────────────────────────────

def run(size: int = 1000,
        days: int = 365,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        mu: float = DEFAULT_MU,
        seed: int = 42,
        snapshot_interval: int = 0,
        verbose: bool = True) -> dict:
    """
    Ejecuta la simulación secuencial completa.

    Parámetros
    ----------
    size              : lado de la grilla cuadrada
    days              : días a simular
    beta              : prob. de contagio por vecino infectado
    gamma             : prob. de recuperación diaria
    mu                : prob. de muerte diaria
    seed              : semilla para reproducibilidad
    snapshot_interval : guardar snapshot cada N días (0 = nunca)
    verbose           : imprimir progreso mensual

    Retorna
    -------
    dict con claves:
        'stats'      : pd.DataFrame  (day, S, I, R, D, cumulative_I)
        'snapshots'  : list[np.ndarray]
        'wall_time'  : float  (segundos)
        'final_grid' : np.ndarray
    """
    rng  = np.random.default_rng(seed)
    grid = initialize_grid(size, seed)

    stats     = []
    snapshots = []
    cum_inf   = int((grid == I).sum())

    t0 = time.perf_counter()

    for day in range(days):
        s_count = int((grid == S).sum())
        i_count = int((grid == I).sum())
        r_count = int((grid == R).sum())
        d_count = int((grid == D).sum())

        stats.append(dict(day=day,
                          S=s_count, I=i_count, R=r_count, D=d_count,
                          cumulative_I=cum_inf))

        if snapshot_interval > 0 and day % snapshot_interval == 0:
            snapshots.append(grid.copy())

        if verbose and day % 30 == 0:
            print(f"  Día {day:3d}: S={s_count:>8,}  I={i_count:>7,}  "
                  f"R={r_count:>8,}  D={d_count:>7,}")

        prev_i = i_count
        grid   = step(grid, beta, gamma, mu, rng)
        new_i  = int((grid == I).sum())
        cum_inf += max(0, new_i - prev_i)

    wall_time = time.perf_counter() - t0

    if verbose:
        print(f"\n  ✓ Secuencial terminado en {wall_time:.2f} s")

    return dict(stats=pd.DataFrame(stats),
                snapshots=snapshots,
                wall_time=wall_time,
                final_grid=grid)


# ── Validación ────────────────────────────────────────────────────────────────

def validate_small(size: int = 50, days: int = 30) -> bool:
    """
    Verificación rápida del modelo:
    el número de infectados debe tener un pico y luego descender.
    Devuelve True si la prueba pasa.
    """
    res = run(size=size, days=days, beta=0.10, gamma=0.15, mu=0.005,
              verbose=False, snapshot_interval=0)
    df       = res['stats']
    peak_day = int(df['I'].idxmax())
    ok       = (peak_day > 1) and (df['I'].iloc[-1] < df['I'].iloc[peak_day])
    status   = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Validación [{status}]: pico de infectados en día {peak_day}, "
          f"termina con {df['I'].iloc[-1]} infectados")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Validación (50×50, 30 días) ===")
    validate_small()

    print("\n=== Simulación completa (1 000×1 000, 365 días) ===")
    result = run(snapshot_interval=7, verbose=True)
    df = result['stats']
    print(f"\n  Pico de infectados : {df['I'].max():>8,}  (día {df['I'].idxmax()})")
    print(f"  Infectados totales : {df['cumulative_I'].iloc[-1]:>8,}")
    print(f"  Muertos totales    : {df['D'].iloc[-1]:>8,}")
    print(f"  Tiempo de pared    : {result['wall_time']:.2f} s")

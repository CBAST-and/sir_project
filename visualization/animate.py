"""
animate.py
──────────
Genera animaciones GIF/MP4 del brote epidémico,
mostrando secuencial vs. paralelo side-by-side.

Uso:
    python visualization/animate.py [--fps 8] [--out animation.gif]

Requiere que main.py ya haya guardado snapshots en snapshots/.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SNAP_DIR = ROOT / "snapshots"

# Mapa de colores: S=azul, I=rojo, R=verde, D=negro
CMAP = ListedColormap(["#3B82F6",   # Susceptible  – azul
                       "#EF4444",   # Infectado    – rojo
                       "#22C55E",   # Recuperado   – verde
                       "#1C1C1E"])  # Muerto        – casi negro
NORM = BoundaryNorm([0, 1, 2, 3, 4], CMAP.N)

LEGEND_HANDLES = [
    Patch(facecolor="#3B82F6", label="Susceptible (S)"),
    Patch(facecolor="#EF4444", label="Infectado (I)"),
    Patch(facecolor="#22C55E", label="Recuperado (R)"),
    Patch(facecolor="#1C1C1E", label="Muerto (D)"),
]


# ── I/O de snapshots ──────────────────────────────────────────────────────────

def save_snapshots(result: dict, tag: str) -> None:
    """Guarda los snapshots de un resultado en snapshots/<tag>/snap_NNNN.npy"""
    d = SNAP_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    for k, snap in enumerate(result['snapshots']):
        np.save(d / f"snap_{k:04d}.npy", snap)
    print(f"  {len(result['snapshots'])} snapshots guardados → {d}")


def load_snapshots(tag: str) -> list:
    """Carga todos los snapshots de snapshots/<tag>/"""
    d = SNAP_DIR / tag
    if not d.exists():
        return []
    paths = sorted(d.glob("snap_*.npy"))
    return [np.load(p) for p in paths]


# ── Animación ─────────────────────────────────────────────────────────────────

def animate_side_by_side(snaps_seq: list,
                         snaps_par: list,
                         interval_days: int = 7,
                         out_path: Path = None,
                         fps: int = 8) -> Path:
    """
    Genera animación GIF (y MP4 si hay ffmpeg) con dos paneles:
        Izquierda: simulación secuencial
        Derecha  : simulación paralela

    Parámetros
    ----------
    snaps_seq      : lista de grillas (secuencial)
    snaps_par      : lista de grillas (paralela)
    interval_days  : días entre snapshots consecutivos
    out_path       : ruta de salida (sin extensión); default = ROOT/animation
    fps            : fotogramas por segundo

    Retorna la ruta al GIF generado.
    """
    if out_path is None:
        out_path = ROOT / "animation"

    n_frames = min(len(snaps_seq), len(snaps_par))
    if n_frames == 0:
        print("  No se encontraron snapshots. Ejecuta main.py primero.")
        return None

    size = snaps_seq[0].shape[0]

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 6.5), facecolor="#111827")
    fig.suptitle("Dispersión del Brote Epidémico — SIR 2D",
                 color="white", fontsize=14, fontweight='bold', y=0.97)

    ax_seq = fig.add_axes([0.03, 0.12, 0.44, 0.80])
    ax_par = fig.add_axes([0.53, 0.12, 0.44, 0.80])

    for ax, title in [(ax_seq, "Secuencial"), (ax_par, "Paralelo")]:
        ax.set_facecolor("#111827")
        ax.set_title(title, color="white", fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    im_seq = ax_seq.imshow(snaps_seq[0], cmap=CMAP, norm=NORM,
                            interpolation="nearest", animated=True)
    im_par = ax_par.imshow(snaps_par[0], cmap=CMAP, norm=NORM,
                            interpolation="nearest", animated=True)

    # ── Leyenda ───────────────────────────────────────────────────────────────
    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=4,
               fontsize=9.5, framealpha=0.15, labelcolor="white",
               facecolor="#1F2937", edgecolor="#374151",
               bbox_to_anchor=(0.5, 0.01))

    # ── Texto del día ─────────────────────────────────────────────────────────
    day_text = fig.text(0.5, 0.92, "", ha="center", va="top",
                        color="#D1D5DB", fontsize=11)

    # ── Barra de progreso (simple rect) ──────────────────────────────────────
    bar_bg  = fig.add_axes([0.10, 0.055, 0.80, 0.012])
    bar_fg  = fig.add_axes([0.10, 0.055, 0.0,  0.012])
    for ax in [bar_bg, bar_fg]:
        ax.set_xticks([]); ax.set_yticks([])
    bar_bg.set_facecolor("#374151")
    bar_fg.set_facecolor("#3B82F6")

    def update(frame: int):
        im_seq.set_data(snaps_seq[frame % len(snaps_seq)])
        im_par.set_data(snaps_par[frame % len(snaps_par)])
        day_text.set_text(f"Día {frame * interval_days:>4d}")
        # Actualizar barra de progreso
        frac = frame / max(n_frames - 1, 1)
        bar_fg.set_position([0.10, 0.055, 0.80 * frac, 0.012])
        return [im_seq, im_par, day_text]

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=True)

    # ── Guardar GIF ───────────────────────────────────────────────────────────
    gif_path = Path(str(out_path) + ".gif")
    print(f"  Guardando GIF → {gif_path} …")
    writer_gif = animation.PillowWriter(fps=fps)
    ani.save(str(gif_path), writer=writer_gif)
    print(f"  ✓ GIF guardado ({gif_path.stat().st_size / 1e6:.1f} MB)")

    # ── Intentar MP4 ─────────────────────────────────────────────────────────
    mp4_path = Path(str(out_path) + ".mp4")
    try:
        writer_mp4 = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                            extra_args=['-vcodec', 'libx264'])
        ani.save(str(mp4_path), writer=writer_mp4)
        print(f"  ✓ MP4 guardado  → {mp4_path}")
    except Exception as e:
        print(f"  ⚠ MP4 omitido ({e}). El GIF está disponible.")

    plt.close()
    return gif_path


# ── Plot de curvas epidémicas (bonus) ─────────────────────────────────────────

def plot_epidemic_curves(stats_seq, stats_par,
                         out_path: Path = None) -> None:
    """
    Grafica las curvas S, I, R, D para ambas simulaciones.
    """
    if out_path is None:
        out_path = ROOT / "data" / "epidemic_curves.png"
    out_path.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Curvas Epidémicas — SIR 2D (1 000×1 000)", fontsize=13)

    for ax, df, title in [(axes[0], stats_seq, "Secuencial"),
                          (axes[1], stats_par, "Paralelo")]:
        ax.plot(df['day'], df['S'], color="#3B82F6", lw=1.8, label="Susceptibles")
        ax.plot(df['day'], df['I'], color="#EF4444", lw=2.0, label="Infectados")
        ax.plot(df['day'], df['R'], color="#22C55E", lw=1.8, label="Recuperados")
        ax.plot(df['day'], df['D'], color="#9CA3AF", lw=1.5, label="Muertos",
                linestyle="--")
        ax.set_xlabel("Día")
        ax.set_ylabel("Personas")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Curvas epidémicas → {out_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animar brote SIR")
    parser.add_argument("--fps",      type=int,   default=8)
    parser.add_argument("--interval", type=int,   default=7,
                        help="Días entre snapshots consecutivos")
    parser.add_argument("--out",      type=str,   default=str(ROOT / "animation"))
    args = parser.parse_args()

    print("Cargando snapshots …")
    snaps_seq = load_snapshots("sequential")
    snaps_par = load_snapshots("parallel")

    if not snaps_seq or not snaps_par:
        print("No se encontraron snapshots. Ejecuta main.py primero.")
        sys.exit(1)

    animate_side_by_side(snaps_seq, snaps_par,
                         interval_days=args.interval,
                         out_path=Path(args.out),
                         fps=args.fps)

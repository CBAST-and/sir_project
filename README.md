# Simulación SIR 2D — Paralela vs Secuencial

Modelo epidemiológico SIR estocástico en una grilla 2-D de 1 000 × 1 000 celdas
(1 M de personas). Comparación de rendimiento entre implementación secuencial y
paralela con `multiprocessing` + ghost cells.

---

## Estructura del proyecto

```
sir_project/
├── sequential/
│   └── sir_sequential.py      ← modelo secuencial + validación
├── parallel/
│   └── sir_parallel.py        ← versión paralela (ghost cells)
├── experiments/
│   └── run_scaling.py         ← strong scaling 1–8 cores
├── visualization/
│   └── animate.py             ← GIF/MP4 side-by-side
├── data/                      ← CSVs y gráficas (generado al correr)
├── snapshots/                 ← grillas guardadas (generado al correr)
├── main.py                    ← pipeline completo
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
# 1. Clonar / descargar el proyecto
cd sir_project

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso rápido

```bash
# Prueba rápida (200×200, 60 días, ~30 s)
python main.py --quick

# Simulación completa (1 000×1 000, 365 días)
python main.py

# Sólo validación + secuencial + paralelo, sin scaling ni animación
python main.py --skip-scaling --skip-animation

# Correr sólo el experimento de scaling
python experiments/run_scaling.py --size 1000 --days 365

# Correr sólo la animación (requiere snapshots previos)
python visualization/animate.py
```

---

## Modelo matemático

### 1. Estados

Cada celda $(i, j)$ en la grilla $\mathcal{G} = \{0,\ldots,N-1\}^2$ se encuentra en
uno de cuatro estados discre­tos en el tiempo $t$:

| Símbolo | Valor | Descripción |
|---------|-------|-------------|
| S | 0 | Susceptible |
| I | 1 | Infectado / infeccioso |
| R | 2 | Recuperado / inmune |
| D | 3 | Muerto |

### 2. Vecindad

Se usa la **vecindad de Moore** de radio 1, es decir los 8 vecinos ortogonales y
diagonales de cada celda. La grilla es **toroidal** (los bordes se conectan entre sí).

Sea $k_{ij}^t$ el número de vecinos infectados de la celda $(i,j)$ en el día $t$:

$$k_{ij}^t = \sum_{(di,dj)\in\mathcal{N}\setminus\{(0,0)\}} \mathbf{1}\!\left[X_{i+di,j+dj}^t = I\right]$$

donde $\mathcal{N} = \{-1,0,1\}^2$.

### 3. Reglas de transición (por celda, por día)

$$P(S \to I \mid k_{ij}^t) = 1 - (1-\beta)^{k_{ij}^t}$$

> *Probabilidad de que al menos uno de los $k$ contactos infectados contagie.*

$$P(I \to R) = \gamma$$

$$P(I \to D \mid \text{no recuperado}) = \mu$$

Las transiciones $I \to R$ y $I \to D$ son mutuamente excluyentes en el mismo paso:
primero se evalúa la recuperación; sólo si no ocurre se evalúa la muerte.

### 4. Número reproductivo básico

En el modelo espacial discreto, el $R_0$ efectivo depende de la densidad local de
susceptibles, pero la aproximación de campo medio da:

$$R_0 \approx \frac{8\,\beta}{\gamma + \mu}$$

Con los parámetros por defecto ($\beta=0.05$, $\gamma=0.10$, $\mu=0.005$):

$$R_0 \approx \frac{8 \times 0.05}{0.10 + 0.005} = \frac{0.40}{0.105} \approx 3.8$$

### 5. Parámetros por defecto

| Parámetro | Valor | Interpretación |
|-----------|-------|----------------|
| β | 0.05 | Prob. de contagio por vecino infectado por día |
| γ | 0.10 | Prob. de recuperación diaria (~10 días promedio) |
| μ | 0.005 | Prob. de muerte diaria (~1 % de fatali­dad) |
| Grilla | 1 000×1 000 | 1 000 000 personas |
| Días | 365 | 1 año calendario |
| Semilla | 42 | Reproducibilidad |

---

## Paralelización

### Estrategia: franja horizontal + ghost cells

La grilla se divide en **N franjas horizontales** (una por worker):

```
┌─────────────────────┐
│  ghost row superior │  ← copia de la última fila de la franja k-1
├─────────────────────┤
│                     │
│    franja real k    │  ← procesada por el worker k
│                     │
├─────────────────────┤
│  ghost row inferior │  ← copia de la primera fila de la franja k+1
└─────────────────────┘
```

Cada worker recibe su franja + las ghost rows, calcula la convolución de vecinos
infectados con `scipy.ndimage.convolve(mode='constant')`, y devuelve **sólo las
filas reales** actualizadas.

El proceso principal re-ensambla la grilla con `np.concatenate` y calcula las
estadísticas globales (reducción paralela).

### Reducción de estadísticas

Los conteos globales de S, I, R, D se calculan sobre la grilla ensamblada
con operaciones vectorizadas de NumPy. En una implementación con memoria
distribuida, esto equivaldría a una reducción MPI `MPI_SUM`.

---

## Salidas generadas

| Archivo | Descripción |
|---------|-------------|
| `data/stats_sequential.csv` | Serie temporal S, I, R, D (secuencial) |
| `data/stats_parallel.csv`   | Serie temporal S, I, R, D (paralelo) |
| `data/scaling_results.csv`  | Tiempos y speed-up por número de cores |
| `data/speedup.png`          | Gráfica de speed-up, eficiencia y tiempo |
| `data/epidemic_curves.png`  | Curvas epidémicas comparativas |
| `animation.gif`             | Animación side-by-side (GIF) |
| `animation.mp4`             | Animación side-by-side (MP4, requiere ffmpeg) |

---

## Dependencias

| Paquete | Uso |
|---------|-----|
| `numpy` | Aritmética vectorizada de grillas |
| `scipy` | `ndimage.convolve` para conteo de vecinos |
| `pandas` | Almacenamiento de estadísticas |
| `matplotlib` | Gráficas y animaciones |
| `Pillow` | Escritura de GIFs |

Python ≥ 3.9 recomendado.
`multiprocessing` es de la biblioteca estándar (no requiere instalación extra).

---

## Notas de implementación

* **Semilla aleatoria**: la versión paralela usa semillas deterministas por
  `(día × 9973 + worker × 1 000 003) mod 2^32`, garantizando reproducibilidad.
  Los resultados cuantitativos difieren levemente de la versión secuencial
  (expectado: los RNG son independientes por franja).

* **Grilla toroidal**: la versión secuencial usa `mode='wrap'` en la convolución;
  la paralela usa `mode='constant'` con las ghost cells para el mismo efecto
  sin artefactos en los bordes internos.

* **Windows**: se incluye `mp.freeze_support()` en `main.py` para compatibilidad
  con el mecanismo de `spawn` de Windows.

---

*Proyecto académico — Sistemas Paralelos y Distribuidos*

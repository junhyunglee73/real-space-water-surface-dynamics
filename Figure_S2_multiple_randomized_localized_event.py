import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration
# ========================================

WIDTH  = 300
HEIGHT = 480

dx = 1.0
dt = 0.8
T_steps = 1000

# Grid (static geometry)
x_vals = np.linspace(0, WIDTH - 1, WIDTH)
y_vals = np.linspace(0, HEIGHT - 1, HEIGHT)
X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")

# ========================================
# Distributed Initial Sources (Rain-on-Floor)
# ========================================

cx_center = (WIDTH  - 1) / 2.0
cy_center = (HEIGHT - 1) / 2.0

sources = []

grid_cols = 2
grid_rows = 3

spacing_x = WIDTH / grid_cols
spacing_y = HEIGHT / grid_rows

start_x = spacing_x / 2
start_y = spacing_y / 2

np.random.seed(42)

for r in range(grid_rows):
    for c in range(grid_cols):

        jitter_x = np.random.randint(-50, 50)
        jitter_y = np.random.randint(-50, 50)

        pos_x = np.clip(start_x + c * spacing_x + jitter_x, 0, WIDTH - 1)
        pos_y = np.clip(start_y + r * spacing_y + jitter_y, 0, HEIGHT - 1)

        rand_step = np.random.randint(0, 50)
        start_t = rand_step * dt

        sources.append({
            "pos": (pos_x, pos_y),
            "start_time": start_t,
            "force": 4.0,
            "type": "origin"
        })


# ========================================
# Boundary Causal State
# ========================================

boundary_triggered = {
    "left": False,
    "right": False,
    "bottom": False,
    "top": False
}

# ========================================
# Physical Parameters (Invariant Law)
# ========================================

stiffness_k = 0.5
mass_m = 50.0
damping_c = 1.0
V_record = 1.0

gamma = damping_c / (2 * mass_m)
w_sq  = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_sq) if w_sq > 0 else 0.0

# ========================================
# Expanded Causal Horizon Operators
# ========================================

def H_time(tau):
    """Temporal admissibility of information."""
    return (tau >= 0).astype(float)

def H_space():
    """Spatial admissibility of information."""
    return 1.0

def hookean_state_operator(tau):
    """Local deterministic state operator."""
    return np.cos(natural_freq * tau)

# ========================================
# Unified Governing Kernel
# ========================================

def causal_record_kernel(t_current, src):

    local_time = t_current - src["start_time"]
    if local_time < 0:
        return np.zeros((HEIGHT, WIDTH))

    sx, sy = src["pos"]
    r = np.sqrt((X - sx)**2 + (Y - sy)**2)

    tau = local_time - (r / V_record)

    H_causal = H_time(tau) * H_space()
    decay = np.exp(-gamma * tau)
    state = hookean_state_operator(tau)

    return src["force"] * decay * state * H_causal

# ========================================
# Time Evolution & Boundary Regeneration
# ========================================

snap_surface_2d = []
snap_cross_1d   = []
capture_times  = []

capture_steps = [50, 200, 400, 600, 800]

base_force = sources[0]["force"]

for step in range(T_steps + 1):
    current_time = step * dt
    propagation_radius = V_record * current_time

    # --- Boundary as causal regenerators ---

    if (not boundary_triggered["left"]) and (propagation_radius >= cx_center):
        boundary_triggered["left"] = True
        sources.append({
            "pos": (0.0, cy_center),
            "start_time": current_time,
            "force": base_force * 0.9,
            "type": "boundary"
        })

    if (not boundary_triggered["right"]) and (propagation_radius >= (WIDTH - 1 - cx_center)):
        boundary_triggered["right"] = True
        sources.append({
            "pos": (WIDTH - 1, cy_center),
            "start_time": current_time,
            "force": base_force * 0.9,
            "type": "boundary"
        })

    if (not boundary_triggered["bottom"]) and (propagation_radius >= cy_center):
        boundary_triggered["bottom"] = True
        sources.append({
            "pos": (cx_center, 0.0),
            "start_time": current_time,
            "force": base_force * 0.9,
            "type": "boundary"
        })

    if (not boundary_triggered["top"]) and (propagation_radius >= (HEIGHT - 1 - cy_center)):
        boundary_triggered["top"] = True
        sources.append({
            "pos": (cx_center, HEIGHT - 1),
            "start_time": current_time,
            "force": base_force * 0.9,
            "type": "boundary"
        })

    # --- Snapshot capture ---
    if step in capture_steps:
        surface = np.zeros((HEIGHT, WIDTH))
        for src in sources:
            surface += causal_record_kernel(current_time, src)

        snap_surface_2d.append(surface.copy())
        snap_cross_1d.append(surface[int(cy_center), :].copy())
        capture_times.append(step)

# ========================================
# Visualization
# ========================================

num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(18, 8), constrained_layout=True)

VISUAL_LIMIT = 5.0
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):

    im = axes[0, i].imshow(
        snap_surface_2d[i],
        cmap="seismic",
        norm=norm,
        origin="lower",
        extent=[0, WIDTH, 0, HEIGHT]
    )
    axes[0, i].set_title(f"t = {capture_times[i]} steps", fontsize=11)

    active = [s for s in sources if s["start_time"] <= capture_times[i] * dt]
    origin = [s for s in active if s["type"] == "origin"]
    boundary = [s for s in active if s["type"] == "boundary"]

    if origin:
        axes[0, i].scatter(
            [s["pos"][0] for s in origin],
            [s["pos"][1] for s in origin],
            c="yellow", marker="x", s=20, alpha=0.5
        )

    if boundary:
        axes[0, i].scatter(
            [s["pos"][0] for s in boundary],
            [s["pos"][1] for s in boundary],
            c="cyan", marker="o", s=40, alpha=0.9
        )

    axes[0, i].axis("off")

    axes[1, i].plot(
        np.arange(WIDTH),
        snap_cross_1d[i],
        color="black",
        linewidth=1.2
    )
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.5, VISUAL_LIMIT * 1.5)
    axes[1, i].set_title(f"Cross-section y = {int(cy_center)}", fontsize=10)

    if i == 0:
        axes[1, i].set_ylabel("Recorded State", fontsize=10)
    else:
        axes[1, i].set_yticklabels([])

cbar = fig.colorbar(im, ax=axes[0, :], location="right", fraction=0.015, pad=0.02)

plt.show()

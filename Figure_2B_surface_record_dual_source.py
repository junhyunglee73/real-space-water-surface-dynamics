# ------------------------------------------------
# Dual Source Interaction
# Deterministic Causal Record Superposition
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry
# ========================================

HEIGHT, WIDTH = 400, 400
dx = 1.0

dt = 0.5
T_steps = 800

# Grid
y, x = np.ogrid[:HEIGHT, :WIDTH]

# Domain center
cy, cx = HEIGHT // 2, WIDTH // 2

# --- Dual source configuration (horizontal) ---
source_positions = [
    (cx - 100, cy),   # Left source
    (cx + 100, cy)    # Right source
]

# Precompute distance maps (static geometry)
dist_maps = [
    np.sqrt((x - sx)**2 + (y - sy)**2)
    for (sx, sy) in source_positions
]

# ========================================
# Physical Parameters (Causes)
# ========================================

input_force_magnitude = 5.0

stiffness_k = 0.5
mass_m = 50.0
damping_c = 0.6

V_record = 1.0

# --- Derived Hookean state parameters ---
gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Expanded Causal Horizon
# ========================================

def H_time(tau):
    """Temporal admissibility of information."""
    return (tau >= 0).astype(float)


# ========================================
# Hookean State Operator (Non-wave)
# ========================================

def hookean_state_operator(tau):
    """Local deterministic state operator."""
    return np.cos(natural_freq * tau)


# ========================================
# Governing Equation (Causal Record Kernel)
# ========================================

def causal_record_kernel(t_current, r_dist):

    tau = t_current - (r_dist / V_record)

    # Causal horizon
    H_causal = H_time(tau)

    # Hookean state history
    decay = np.exp(-gamma * tau)
    state_sign = hookean_state_operator(tau)

    return decay * state_sign * H_causal


# ========================================
# Time Evolution & Record Superposition
# ========================================

snap_surface_2d = []
snap_cross_1d = []
capture_times = []

capture_steps = [150, 300, 450, 600, 750]

for step in range(T_steps + 1):
    current_time = step * dt

    # --- Superposition of causal records ---
    total_surface_record = np.zeros((HEIGHT, WIDTH))

    for dist_map in dist_maps:
        total_surface_record += (
            input_force_magnitude
            * causal_record_kernel(current_time, dist_map)
        )

    if step in capture_steps:
        snap_surface_2d.append(total_surface_record.copy())
        snap_cross_1d.append(total_surface_record[cy, :].copy())
        capture_times.append(step)
        print(f">> Snapshot captured at step {step}")

# ========================================
# Visualization (Manuscript Style)
# ========================================

num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(
    2, num_snaps,
    figsize=(18, 8),
    constrained_layout=True
)

VISUAL_LIMIT = 5.0
norm = TwoSlopeNorm(
    vmin=-VISUAL_LIMIT,
    vcenter=0,
    vmax=VISUAL_LIMIT
)

for i in range(num_snaps):

    # --- 2D surface record ---
    im1 = axes[0, i].imshow(
        snap_surface_2d[i],
        cmap="seismic",
        norm=norm,
        origin="lower"
    )
    axes[0, i].set_title(
        f"Causal Record Superposition (t={capture_times[i]})",
        fontsize=11
    )
    axes[0, i].axis("off")

    # Mark source positions
    axes[0, i].scatter(
        [p[0] for p in source_positions],
        [p[1] for p in source_positions],
        c="yellow",
        s=40,
        marker="x",
        alpha=0.8
    )

    # --- 1D cross-section ---
    x_axis = np.arange(WIDTH)
    axes[1, i].plot(
        x_axis,
        snap_cross_1d[i],
        color="black",
        linewidth=1.5
    )
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(
        -VISUAL_LIMIT * 1.8,
        VISUAL_LIMIT * 1.8
    )
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    axes[1, i].set_title(
        f"Net Recorded State (t={capture_times[i]})",
        fontsize=11
    )

    if i > 0:
        axes[1, i].set_yticklabels([])
    else:
        axes[1, i].set_ylabel(
            "Recorded State (+ / −)",
            fontsize=10
        )

# Colorbar
cbar = fig.colorbar(
    im1,
    ax=axes[0, :],
    location="right",
    fraction=0.015,
    pad=0.02
)

plt.show()

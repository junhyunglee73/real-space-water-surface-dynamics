# ------------------------------------------------
# Local Deterministic Recording (Single Source)
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry
# ========================================

# Spatial domain (real-space substrate)
HEIGHT, WIDTH = 400, 400
dx = 1.0

# Temporal domain
dt = 0.5
T_steps = 600

# Center of the domain (single source)
cy, cx = HEIGHT // 2, WIDTH // 2

# Grid (static geometry)
y, x = np.ogrid[:HEIGHT, :WIDTH]
dist_map = np.sqrt((x - cx)**2 + (y - cy)**2)

# ========================================
# Physical Parameters (The Causes)
# ========================================

# (A) Input force magnitude (vector scale)
input_force_magnitude = 15

# (B) Hookean local response (state-based)
stiffness_k = 0.5
mass_m = 20.0
damping_c = 0.6

# (C) Causal propagation speed
V_record = 1.0

# --- Derived parameters  ---
gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Horizon 
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
# Governing Equation (Causal Recording Kernel)
# ========================================

def causal_record_kernel(t_current, r_dist):

    # Local causal time
    tau = t_current - (r_dist / V_record)

    # --- Causal Horizon ---
    H_causal = H_time(tau) 

    # --- Hookean state history ---
    decay = np.exp(-gamma * tau)
    state_sign = hookean_state_operator(tau)

    return decay * state_sign * H_causal


# ========================================
# Time Evolution & Snapshot Recording
# ========================================

snap_surface_2d = []   # 2D surface records
snap_cross_1d  = []   # 1D cross-sections
capture_times  = []

# Snapshot schedule (for manuscript figures)
capture_steps = [50, 150, 300, 450, 500]

for step in range(T_steps + 1):
    current_time = step * dt

    # Governing equation (single source)
    surface_record = (
        input_force_magnitude
        * causal_record_kernel(current_time, dist_map)
    )

    # Capture snapshots
    if step in capture_steps:
        snap_surface_2d.append(surface_record.copy())
        snap_cross_1d.append(surface_record[cy, :].copy())
        capture_times.append(step)

# ========================================
# Visualization
# ========================================

num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(
    2, num_snaps,
    figsize=(16, 8),
    constrained_layout=True
)

# Fixed visual scale for consistency
VISUAL_LIMIT = 5.0
norm = TwoSlopeNorm(
    vmin=-VISUAL_LIMIT,
    vcenter=0,
    vmax=VISUAL_LIMIT
)

for i in range(num_snaps):

    # --- Top view: 2D surface record ---
    im1 = axes[0, i].imshow(
        snap_surface_2d[i],
        cmap="seismic",
        norm=norm,
        origin="lower"
    )
    axes[0, i].set_title(
        f"Surface Record (t={capture_times[i]})",
        fontsize=11
    )
    axes[0, i].axis("off")

    # Mark cross-section line
    axes[0, i].axhline(
        cy,
        color="black",
        linestyle="--",
        alpha=0.3,
        linewidth=1
    )

    # --- Bottom view: 1D cross-section ---
    x_axis = np.arange(WIDTH)
    axes[1, i].plot(
        x_axis,
        snap_cross_1d[i],
        color="black",
        linewidth=1.5
    )
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(
        -VISUAL_LIMIT * 3,
        VISUAL_LIMIT * 3
    )
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    axes[1, i].set_title(
        f"Vector Magnitude Profile (t={capture_times[i]})",
        fontsize=11
    )

    if i > 0:
        axes[1, i].set_yticklabels([])
    else:
        axes[1, i].set_ylabel(
            "Recorded State (+ / −)",
            fontsize=10
        )

# Colorbar (shared)
cbar = fig.colorbar(
    im1,
    ax=axes[0, :],
    location="right",
    fraction=0.015,
    pad=0.02
)

plt.show()

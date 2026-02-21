import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry 
# ========================================
HEIGHT, WIDTH = 850, 850
dx, dt = 1.0, 1
T_steps = 500

cy, cx = HEIGHT // 2, WIDTH // 2
y, x = np.ogrid[:HEIGHT, :WIDTH]
dist_map = np.sqrt((x - cx)**2 + (y - cy)**2)

margin = 50  
beta_field = np.zeros((HEIGHT, WIDTH))

l_wall, r_wall = margin, WIDTH - margin
b_wall, t_wall = margin, HEIGHT - margin
beta_field[b_wall : t_wall, l_wall : r_wall] = 1.0

input_force_magnitude = 5.0
A_field = input_force_magnitude * beta_field 

# ========================================
# Physical Parameters (The Causes)
# ========================================
stiffness_k = 0.5
mass_m = 50.0
damping_c = 0.6
V_record = 1.0

gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Operators & Hookean State Operator
# ========================================
def H_time(tau):
    return (tau >= 0).astype(float)

def hookean_state_operator(tau):
    return np.cos(natural_freq * tau)

# ========================================
# Governing Equation (Causal Recording Kernel)
# ========================================
def causal_record_kernel(t_current, r_dist):
    tau = t_current - (r_dist / V_record)
    H_causal = H_time(tau)
    
    decay = np.exp(-gamma * tau)
    state_sign = hookean_state_operator(tau)
    
    return decay * state_sign * H_causal

# ========================================
# Time Evolution & Snapshot Recording
# ========================================
snap_surface_2d = []
snap_cross_1d   = []
capture_times   = []
capture_steps = [50, 150, 300, 450, 500]

for step in range(T_steps + 1):
    current_time = step * dt

    surface_record = A_field * causal_record_kernel(current_time, dist_map)

    if step in capture_steps:
        snap_surface_2d.append(surface_record.copy())
        snap_cross_1d.append(surface_record[cy, :].copy())
        capture_times.append(step)

# ========================================
# Visualization
# ========================================
num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(16, 8), constrained_layout=True)
VISUAL_LIMIT = 5.0
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    # 2D Surface Plot
    im1 = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"Surface Record (t={capture_times[i]})")
    axes[0, i].axis("off")
    
    # Add boundary rectangle
    rect = plt.Rectangle((l_wall, b_wall), r_wall - l_wall, t_wall - b_wall, 
                         linewidth=1.5, edgecolor='black', facecolor='none', alpha=0.7)
    axes[0, i].add_patch(rect)

    # 1D Cross-section Plot
    axes[1, i].plot(np.arange(WIDTH), snap_cross_1d[i], color="black", linewidth=1.5)
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.2, VISUAL_LIMIT * 1.2)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)

# Colorbar
cbar = fig.colorbar(im1, ax=axes[0, :], location="right", fraction=0.015, pad=0.02)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry 
# ========================================
margin = 50  # wall margin
w_inside = 450  
h_inside = 450

WIDTH = w_inside + (2 * margin)  
HEIGHT = h_inside + (2 * margin) 
dx, dt = 1.0, 4
T_steps = 2000

cy_center, cx_center = HEIGHT // 2, WIDTH // 2
y_coords, x_coords = np.ogrid[:HEIGHT, :WIDTH]

beta_field = np.zeros((HEIGHT, WIDTH))
l_wall, r_wall = margin, WIDTH - margin
b_wall, t_wall = margin, HEIGHT - margin
beta_field[b_wall : t_wall, l_wall : r_wall] = 1.0

# ========================================
# Distributed Initial Sources (Rain-on-Floor)
# ========================================
sources = []
grid_cols, grid_rows = 5, 5
spacing_x = (r_wall - l_wall) / grid_cols
spacing_y = (t_wall - b_wall) / grid_rows

np.random.seed(42)

for r in range(grid_rows):
    for c in range(grid_cols):
        jitter_x = 0
        jitter_y = 0

        pos_x = np.clip(l_wall + spacing_x/2 + c * spacing_x + jitter_x, l_wall+5, r_wall-5)
        pos_y = np.clip(b_wall + spacing_y/2 + r * spacing_y + jitter_y, b_wall+5, t_wall-5)

        start_t = 0

        sources.append({
            "pos": (pos_x, pos_y),
            "start_time": start_t,
            "force": 5.0,
            "type": "origin",
            "parent_side": None, # Tracking for chain reaction
            "triggered": {"left": False, "right": False, "bottom": False, "top": False}
        })

# ========================================
# Physical Parameters 
# ========================================
stiffness_k = 0.5
mass_m = 100.0
damping_c = 0
V_record = 1.0

gamma = damping_c / (2 * mass_m)
w_sq = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_sq) if w_sq > 0 else 0.0

# ========================================
# Governing Kernel
# ========================================
def causal_record_kernel(t_current, src):
    local_time = t_current - src["start_time"]
    if local_time < 0: return np.zeros((HEIGHT, WIDTH))

    sx, sy = src["pos"]
    r = np.sqrt((x_coords - sx)**2 + (y_coords - sy)**2)
    tau = local_time - (r / V_record)

    H_causal = (tau >= 0).astype(float)
    decay = np.exp(-gamma * tau)
    state = np.cos(natural_freq * tau)

    return src["force"] * beta_field * decay * state * H_causal 

def get_dist_to_walls(pos):
    return {
        "left": pos[0] - l_wall,
        "right": r_wall - pos[0],
        "bottom": pos[1] - b_wall,
        "top": t_wall - pos[1]
    }

# ========================================
# Time Evolution & Snapshot Recording
# ========================================
snap_surface_2d = []
snap_cross_1d   = []
capture_times  = []
capture_steps = [500, 750, 1000, 1250, 1500, 1750, 2000] 

for step in range(T_steps + 1):
    current_time = step * dt * beta_field[cy_center, cx_center]  # Modulate time progression by beta at the center
    
    # --- Infinite Boundary Regeneration (Chain Reaction) ---
    new_borns = []
    for src in sources:
        dists = get_dist_to_walls(src["pos"])
        prop_radius = V_record * (current_time - src["start_time"])
        
        for side, d in dists.items():
            # Trigger if front reaches wall and haven't triggered this side for this source yet
            if d > 0 and prop_radius >= d and not src["triggered"][side]:
                if src["parent_side"] == side: continue 
                
                src["triggered"][side] = True
                sx, sy = src["pos"]
                
                # Point-of-contact regeneration
                if side == "left":    new_pos = (l_wall, sy)
                elif side == "right":   new_pos = (r_wall, sy)
                elif side == "bottom":  new_pos = (sx, b_wall)
                elif side == "top":     new_pos = (sx, t_wall)
                
                opposite = {"left":"right", "right":"left", "bottom":"top", "top":"bottom"}
                
                new_borns.append({
                    "pos": new_pos,
                    "start_time": current_time,
                    "force": src["force"] * 0.5, # Reflection loss
                    "type": "boundary",
                    "parent_side": opposite[side],
                    "triggered": {"left": False, "right": False, "bottom": False, "top": False}
                })
    
    sources.extend(new_borns)

    # --- Snapshot capture ---
    if step in capture_steps:
        surface = np.sum([causal_record_kernel(current_time, s) for s in sources], axis=0)
        snap_surface_2d.append(surface.copy())
        snap_cross_1d.append(surface[cy_center, :].copy())
        capture_times.append(step)

# ========================================
# Visualization
# ========================================
num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(18, 8), constrained_layout=True)
VISUAL_LIMIT = 10.0
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    im = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"t = {capture_times[i]} steps\nSrc: {len(sources)}", fontsize=10)
    
    # Add wall boundaries
    rect = plt.Rectangle((l_wall, b_wall), r_wall - l_wall, t_wall - b_wall, 
                         linewidth=1, edgecolor='black', facecolor='none', alpha=0.5)
    axes[0, i].add_patch(rect)
    axes[0, i].axis("off")

    axes[1, i].plot(np.arange(WIDTH), snap_cross_1d[i], color="black", linewidth=1.2)
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.5, VISUAL_LIMIT * 1.5)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    
# Colorbar
cbar = fig.colorbar(
    im,
    ax=axes[0, :],
    location="right",
    fraction=0.015,
    pad=0.02
)

plt.show()
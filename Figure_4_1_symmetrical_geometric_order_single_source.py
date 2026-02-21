import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry 
# ========================================
margin = 10  # wall margin
w_inside = 800
h_inside = 800

WIDTH = w_inside + (2 * margin)  
HEIGHT = h_inside + (2 * margin) 
dx, dt = 1.0, 4
T_steps = 2000

# position of walls
l_wall, r_wall = margin, WIDTH - margin
b_wall, t_wall = margin, HEIGHT - margin

cy, cx = HEIGHT // 2, WIDTH // 2
y_coords, x_coords = np.ogrid[:HEIGHT, :WIDTH]
beta_field = np.zeros((HEIGHT, WIDTH))

# inner area where beta is 1
beta_field[b_wall : t_wall, l_wall : r_wall] = 1.0

# initial source at the center
input_force_magnitude = 5.0
sources = [{
    "pos": (cx, cy),
    "start_time": 0.0,
    "force": input_force_magnitude,
    "parent_side": None,
    "triggered": {"left": False, "right": False, "bottom": False, "top": False}
}]

# ========================================
# Physical Parameters (The Causes)
# ========================================
stiffness_k = 0.5
mass_m = 100.0
damping_c = 0.0 
V_record = 1.0

gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Operators & Governing Kernel
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
    
    return src["force"] * decay * state * H_causal * beta_field

def get_dist_to_walls(pos):
    return {
        "left": pos[0] - l_wall,
        "right": r_wall - pos[0],
        "bottom": pos[1] - b_wall,
        "top": t_wall - pos[1]
    }

# ========================================
# Time Evolution & Infinite Regeneration
# ========================================
snap_surface_2d = []
snap_cross_1d   = []
capture_times   = []
capture_steps = [500, 750, 1000, 1250, 1500, 1750, 2000] 

for step in range(T_steps + 1):
    current_time = step * dt
    
    # --- Infinite Boundary Regeneration (Chain Reaction) ---
    new_borns = []
    for src in sources:
        dists = get_dist_to_walls(src["pos"])
        prop_radius = V_record * (current_time - src["start_time"])
        
        for side, d in dists.items():
            # Trigger if front reaches wall and haven't triggered this side yet
            if d > 0 and prop_radius >= d and not src["triggered"][side]:
                # Prevent immediate back-triggering to same wall
                if src["parent_side"] == side: continue 
                
                src["triggered"][side] = True
                sx, sy = src["pos"]
                
                # Determine new position at the point of contact
                if side == "left":    new_pos = (l_wall, sy)
                elif side == "right":   new_pos = (r_wall, sy)
                elif side == "bottom":  new_pos = (sx, b_wall)
                elif side == "top":     new_pos = (sx, t_wall)
                
                opposite = {"left":"right", "right":"left", "bottom":"top", "top":"bottom"}
                
                new_borns.append({
                    "pos": new_pos,
                    "start_time": current_time,
                    "force": src["force"] * 0.5, # Reflection loss
                    "parent_side": opposite[side],
                    "triggered": {"left": False, "right": False, "bottom": False, "top": False}
                })
    
    sources.extend(new_borns)

    # --- Snapshot capture ---
    if step in capture_steps:
        total_surface = np.zeros((HEIGHT, WIDTH))
        for src in sources:
            total_surface += causal_record_kernel(current_time, src)
            
        snap_surface_2d.append(total_surface.copy())
        snap_cross_1d.append(total_surface[cy, :].copy())
        capture_times.append(step)

# ========================================
# Visualization
# ========================================
num_snaps = len(snap_surface_2d)
fig, axes = plt.subplots(2, num_snaps, figsize=(18, 8), constrained_layout=True)
VISUAL_LIMIT = input_force_magnitude * 2
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    # 2D surface plot
    im1 = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"t={capture_times[i]}\nSources: {len([s for s in sources if s['start_time'] <= capture_times[i]*dt])}", fontsize=10)
    axes[0, i].axis("off")
    
    # Add wall boundaries
    rect = plt.Rectangle((l_wall, b_wall), r_wall - l_wall, t_wall - b_wall, 
                         linewidth=1, edgecolor='black', facecolor='none', alpha=0.5)
    axes[0, i].add_patch(rect)

    # Cross-sectional plot
    axes[1, i].plot(np.arange(WIDTH), snap_cross_1d[i], color="black", linewidth=1.2)
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.5, VISUAL_LIMIT * 1.5)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    if i == 0:
        axes[1, i].set_ylabel("State Magnitude")

# Colorbar
cbar = fig.colorbar(
    im1,
    ax=axes[0, :],
    location="right",
    fraction=0.015,
    pad=0.02
)

plt.show()
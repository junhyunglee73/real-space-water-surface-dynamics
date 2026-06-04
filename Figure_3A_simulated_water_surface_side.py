import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========================================
# System Configuration & Geometry 
# ========================================
margin = 10
w_inside = 300
h_inside = 480

WIDTH = w_inside + (2 * margin) 
HEIGHT = h_inside + (2 * margin) 
dx, dt = 1.0, 0.8
T_steps = 1000

l_wall, r_wall = margin, WIDTH - margin
b_wall, t_wall = margin, HEIGHT - margin

cx = int(l_wall + (w_inside - 6))  
cy = int(b_wall + (h_inside / 2.0))

y_coords, x_coords = np.ogrid[:HEIGHT, :WIDTH]
beta_field = np.zeros((HEIGHT, WIDTH))
beta_field[b_wall : t_wall, l_wall : r_wall] = 1.0

input_force_magnitude = 5.0
sources = [{
    "pos": (cx, cy),
    "start_time": 0.0,
    "force": input_force_magnitude,
    "parent_side": None, # no parent since this is the initial source
    "triggered": {"left": False, "right": False, "bottom": False, "top": False}
}]

# ========================================
# Physical Parameters
# ========================================
stiffness_k = 0.5
mass_m = 50.0
damping_c = 0.8 
V_record = 1.0

gamma = damping_c / (2 * mass_m)
w_squared = (stiffness_k / mass_m) - gamma**2
natural_freq = np.sqrt(w_squared) if w_squared > 0 else 0.0

# ========================================
# Causal Kernel 
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
# Time Evolution & Chain Regeneration
# ========================================
snap_surface_2d = []
snap_cross_1d   = []
capture_times   = []
capture_steps = [0, 250, 450, 600]

for step in range(T_steps + 1):
   # Modulate time progression by beta at the source location
    current_time = step * dt * beta_field[cy, cx] 
    # Check each source for potential wall interactions and generate new sources if needed
    new_borns = []
    for src in sources:
        dists = get_dist_to_walls(src["pos"])
        prop_radius = V_record * (current_time - src["start_time"])
        
        for side, d in dists.items():
            # 1) positive distance to wall, 2) force has reached the wall, 3) not already triggered from this side
            if d > 0 and prop_radius >= d and not src["triggered"][side]:
                # block self-triggering to prevent infinite loops
                if src["parent_side"] == side: continue 
                
                src["triggered"][side] = True
                sx, sy = src["pos"]
                
                if side == "left":    new_pos = (l_wall, sy)
                elif side == "right":   new_pos = (r_wall, sy)
                elif side == "bottom":  new_pos = (sx, b_wall)
                elif side == "top":     new_pos = (sx, t_wall)
                
                opposite = {"left":"right", "right":"left", "bottom":"top", "top":"bottom"}
                
                new_borns.append({
                    "pos": new_pos,
                    "start_time": current_time,
                    "force": src["force"] * 0.5, 
                    "parent_side": opposite[side],
                    "triggered": {"left": False, "right": False, "bottom": False, "top": False}
                })
    
    sources.extend(new_borns)

    # Snapshot capture
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
VISUAL_LIMIT = 5.0
norm = TwoSlopeNorm(vmin=-VISUAL_LIMIT, vcenter=0, vmax=VISUAL_LIMIT)

for i in range(num_snaps):
    # 2D View
    im = axes[0, i].imshow(snap_surface_2d[i], cmap="seismic", norm=norm, origin="lower")
    axes[0, i].set_title(f"t={capture_times[i]}\nActive Sources: {len([s for s in sources if s['start_time'] <= capture_times[i]*dt])}", fontsize=10)
    
  
    rect = plt.Rectangle((l_wall, b_wall), w_inside, h_inside, linewidth=1.5, edgecolor='black', facecolor='none', alpha=0.7)
    axes[0, i].add_patch(rect)
    
    # Source Markers
    active_sources = [s for s in sources if s['start_time'] <= capture_times[i]*dt]
    axes[0, i].scatter([s['pos'][0] for s in active_sources], [s['pos'][1] for s in active_sources], 
                       color='yellow', marker='x', s=20, alpha=0.6)
    axes[0, i].axis("off")

    # 1D Cross-section
    axes[1, i].plot(np.arange(WIDTH), snap_cross_1d[i], color="black", linewidth=1.2)
    axes[1, i].set_xlim(0, WIDTH)
    axes[1, i].set_ylim(-VISUAL_LIMIT * 1.5, VISUAL_LIMIT * 1.5)
    axes[1, i].grid(True, linestyle=":", alpha=0.5)
    # axes[1, i].set_title(f"Section at y={cy}")

fig.colorbar(im, ax=axes[0, :], location="right", fraction=0.015, pad=0.02)
plt.show()
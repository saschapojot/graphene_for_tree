import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# Parameters from the document
l = 1  # Å

# Define basis vectors (in-plane only for 2D view)
a0 = np.array([l, 0])
a1 = np.array([np.cos(2*np.pi/3) * l, np.sin(2*np.pi/3) * l])

# Atomic positions within one unit cell (only x, y coordinates)
r_C0 = (1/3) * a0 + (2/3) * a1
r_C1 = (2/3) * a0 + (1/3) * a1

# Create figure
fig, ax = plt.subplots(figsize=(12, 11))

# Define tiling range - increased to show more cells
n_tiles_x = 5  # Number of tiles in a0 direction
n_tiles_y = 5  # Number of tiles in a1 direction

# Store all atoms for plotting
all_C0_atoms = []
all_C1_atoms = []

# Draw tiled unit cells
for i in range(-1, n_tiles_x):
    for j in range(-1, n_tiles_y):
        # Calculate origin of this unit cell
        origin = i * a0 + j * a1

        # Unit cell corners
        cell_corners = np.array([
            origin,
            origin + a0,
            origin + a0 + a1,
            origin + a1,
            origin
        ])

        # Draw unit cell (no fill, black edges)
        if 0 <= i < n_tiles_x - 1 and 0 <= j < n_tiles_y - 1:
            unit_cell = Polygon(cell_corners[:-1], fill=False,
                                edgecolor='black', linewidth=1.5, alpha=0.3, zorder=1)
            ax.add_patch(unit_cell)

        # Draw cell edges - highlight the cell at (i=1, j=1)
        ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'k-',
                linewidth=1.5 if (i == 1 and j == 1) else 0.8,
                alpha=1.0 if (i == 1 and j == 1) else 0.4, zorder=2)

        # Calculate atomic positions in this cell
        C0_pos = origin + r_C0
        C1_pos = origin + r_C1

        all_C0_atoms.append(C0_pos)
        all_C1_atoms.append(C1_pos)

# Convert to arrays
all_C0_atoms = np.array(all_C0_atoms)
all_C1_atoms = np.array(all_C1_atoms)

# Draw C0-C1 bonds for all cells
for i in range(len(all_C0_atoms)):
    C0_pos = all_C0_atoms[i]
    # Find nearby C1 atoms and draw bonds
    for C1_pos in all_C1_atoms:
        distance = np.linalg.norm(C0_pos - C1_pos)
        if distance < l * 0.6:  # Only draw bonds shorter than this threshold
            ax.plot([C0_pos[0], C1_pos[0]], [C0_pos[1], C1_pos[1]],
                    'black', linewidth=1.5, linestyle='-', zorder=3, alpha=0.3)

# Plot all atoms (changed label text to C0 and C1)
ax.scatter(all_C0_atoms[:, 0], all_C0_atoms[:, 1], s=400, c='lightblue',
           edgecolors='darkblue', linewidths=2, marker='o', label='C0', zorder=10)
ax.scatter(all_C1_atoms[:, 0], all_C1_atoms[:, 1], s=400, c='lightblue',
           edgecolors='darkblue', linewidths=2, marker='o', label='C1', zorder=10)

# ---------------------------------------------------------
# NEW CODE ADDED HERE: Label ALL atoms with C0 or C1
# ---------------------------------------------------------
for x, y in all_C0_atoms:
    ax.text(x, y, 'C0', ha='center', va='center', fontsize=9,
            fontweight='bold', color='darkred', zorder=11)

for x, y in all_C1_atoms:
    ax.text(x, y, 'C1', ha='center', va='center', fontsize=9,
            fontweight='bold', color='darkblue', zorder=11)
# ---------------------------------------------------------

# Highlight the primitive unit cell (i=1, j=1) - shifted up and right
primitive_origin = a0 + a1
primitive_cell_corners = np.array([
    primitive_origin,
    primitive_origin + a0,
    primitive_origin + a0 + a1,
    primitive_origin + a1,
    primitive_origin
])
ax.plot(primitive_cell_corners[:, 0], primitive_cell_corners[:, 1], 'k-',
        linewidth=3, label='primitive unit cell', zorder=5)

# Draw basis vectors for the primitive cell - changed to black, starting from new origin
arrow_props = dict(head_width=0.2, head_length=0.15, linewidth=3, zorder=8)
ax.arrow(primitive_origin[0], primitive_origin[1], a0[0], a0[1], fc='black', ec='black',
         **arrow_props, alpha=0.8)
ax.arrow(primitive_origin[0], primitive_origin[1], a1[0], a1[1], fc='black', ec='black',
         **arrow_props, alpha=0.8)

# Add basis vector labels - changed to black
ax.text(primitive_origin[0] + a0[0]/2, primitive_origin[1] - 0.5, r'$\mathbf{a}_0$', fontsize=16, color='black',
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(primitive_origin[0] + a1[0]/2 - 0.4, primitive_origin[1] + a1[1]/2, r'$\mathbf{a}_1$', fontsize=16,
        color='black', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Note: The specific labels for the primitive cell atoms are commented out below
# because the global loop above now covers them.
primitive_r_C0 = primitive_origin + r_C0
primitive_r_C1 = primitive_origin + r_C1
# ax.text(primitive_r_C0[0], primitive_r_C0[1], 'C0', ha='center', va='center', fontsize=12,
#         fontweight='bold', color='darkred', zorder=11)
# ax.text(primitive_r_C1[0], primitive_r_C1[1], 'C1', ha='center', va='center', fontsize=12,
#         fontweight='bold', color='darkblue', zorder=11)

# Draw circle centered at C0 in primitive cell with radius sqrt(3)*l
circle_radius = np.sqrt(3) * l
circle = Circle((primitive_r_C0[0], primitive_r_C0[1]), circle_radius,
                fill=False, edgecolor='red', linewidth=2,
                linestyle='--', alpha=0.7, zorder=4,
                label=f'Circle centered at C0 (r=√3l={circle_radius:.3f} Å)')
ax.add_patch(circle)

# Add grid for reference
ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, zorder=0)

# Set axis properties
ax.set_xlabel('x (Å)', fontsize=14, fontweight='bold')
ax.set_ylabel('y (Å)', fontsize=14, fontweight='bold')
ax.set_title('Tiling of 2D Hexagonal Unit Cells: graphene (Space Group 191)',
             fontsize=16, fontweight='bold', pad=20)

# Set equal aspect ratio
ax.set_aspect('equal')

# Set axis limits
padding = 1.0
x_min = -1.5 * l
x_max = (n_tiles_x - 0.5) * l
y_min = -1.5 * l
y_max = (n_tiles_y - 0.5) * l * np.sin(2*np.pi/3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Add legend
ax.legend(loc='best', fontsize=12, framealpha=0.95,
          edgecolor='black', fancybox=True)

# Add text box with information - changed to white background
textstr = f'Lattice parameter: a = {l} Å\n'
textstr += f'Angle: γ = 120°\n'
textstr += f'C0-C1 distance: {np.linalg.norm(r_C0 - r_C1):.4f} Å\n'
textstr += f'Unit cell area: {abs(np.cross(a0, a1)):.3f} Å²\n'
textstr += f'Tiles shown: {(n_tiles_x-1)} × {(n_tiles_y-1)}\n'
textstr += f'Circle radius: √3l = {circle_radius:.4f} Å'

props = dict(boxstyle='round', facecolor='white', alpha=0.95,
             edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('graphene_tiling_circle_at_C0_labeled.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print information
print("Hexagonal Tiling Information")
print("=" * 60)
print(f"\nLattice parameter: a = {l} Å")
print(f"Angle between basis vectors: γ = 120°")
print(f"C0-C1 bond length: {np.linalg.norm(r_C0 - r_C1):.4f} Å")
print(f"Unit cell area: {abs(np.cross(a0, a1)):.4f} Å²")
print(f"\nNumber of unit cells shown: {(n_tiles_x-1) * (n_tiles_y-1)}")
print(f"Total atoms shown: {len(all_C0_atoms)} C0 atoms, {len(all_C1_atoms)} C1 atoms")
print(f"\nThe primitive unit cell is highlighted in black")
print(f"Each unit cell contains 1 C0 atom and 1 C1 atom")
print(f"\nCircle centered at C0 with radius √3·l = {circle_radius:.4f} Å")
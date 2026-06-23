"""
HBML Framework Flowchart - Temporal Dependencies Version
Shows the flow of information across time steps with clear dependencies
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure with high DPI for quality
fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Define color scheme
colors = {
    'input': '#3498db',      # Blue - CGM input
    'expert': '#e74c3c',     # Red - Expert models
    'hedge': '#9b59b6',      # Purple - Hedge algorithm
    'output': '#27ae60',     # Green - Forecast output
    'loss': '#f39c12',       # Orange - Loss computation
    'text': '#2c3e50',       # Dark gray
    'light_bg': '#ecf0f1'    # Light gray
}

def add_box(ax, x, y, width, height, text, color, textcolor='white', fontsize=10, bold=True):
    """Add a rounded rectangle box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.08", 
        facecolor=color, 
        edgecolor='white',
        linewidth=2,
        zorder=2
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, color=textcolor, weight=weight, zorder=3)
    return box

def add_straight_arrow(ax, x1, y1, x2, y2, color='#34495e', linewidth=2):
    """Add a straight arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        color=color, 
        linewidth=linewidth,
        zorder=1,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(8, 5.7, 'Hedge-Based Machine Learning (HBML) Framework', 
        ha='center', fontsize=16, weight='bold', color=colors['text'])

# Time arrow at top
ax.annotate('', xy=(15.5, 5.2), xytext=(0.5, 5.2),
            arrowprops=dict(arrowstyle='->', lw=3, color=colors['input']))
ax.text(0.2, 5.2, 'Time', ha='right', fontsize=11, color=colors['input'], 
        weight='bold', va='center')

# Define time step positions - increased spacing for full width
time_positions = [1, 5, 9, 13, 17]
time_labels = ['t-2', 't-1', 't', 't+1', 't+2']

# Vertical positions
y_input = 4.5 + 0.7       # CGM input
y_forecast = 3.3 + 0.7    # Forecast output
y_loss = 2.1 + 0.7        # Loss computation
y_expert = 3.5 + 0.7      # Expert models (between time steps)
y_hedge = 2.1 + 0.7       # Hedge algorithm (between time steps)

# Draw time step columns
for i, (x_pos, label) in enumerate(zip(time_positions, time_labels)):
    # 1. Blue box: CGM input
    add_box(ax, x_pos, y_input, 1.0, 0.5, f'$y_{{{label}}}$', 
            colors['input'], fontsize=11)
    
    # 2. Green box: Forecast (directly below input)
    add_box(ax, x_pos, y_forecast, 1.0, 0.5, f'$\\hat{{y}}_{{{label}}}$', 
            colors['output'], fontsize=11)
    
    # 3. Orange box: Loss (directly below forecast)
    add_box(ax, x_pos, y_loss, 1.0, 0.5, f'$\\ell_{{{label}}}$', 
            colors['loss'], fontsize=10)
    
    # Arrow from input to forecast (vertical)
    # We'll add this later with the processing blocks

# Draw processing blocks between time steps
for i in range(len(time_positions) - 1):
    x_left = time_positions[i]
    x_right = time_positions[i + 1]
    x_between = (x_left + x_right) / 2
    
    # Red box: Expert models
    add_box(ax, x_between, y_expert, 1.2, 0.6, 
            'Experts\n$\\hat{f}_{1:K}$', 
            colors['expert'], fontsize=9)
    
    # Purple box: Hedge algorithm
    add_box(ax, x_between, y_hedge, 1.2, 0.6, 
            'Hedge\nAlgorithm', 
            colors['hedge'], fontsize=9)
    
    # Arrows coming into Expert box
    # From orange loss box on the left to Experts
    add_straight_arrow(ax, x_left + 0.5, y_loss, x_between - 0.7, y_expert,
                      color=colors['loss'], linewidth=1.8)

    # From orange loss box on the left to Hedge
    add_straight_arrow(ax, x_left + 0.5, y_loss, x_between - 0.7, y_hedge,
                      color=colors['loss'], linewidth=1.8)
    
    # From blue input box on the left
    add_straight_arrow(ax, x_left + 0.5, y_input, x_between - 0.7, y_expert,
                      color=colors['input'], linewidth=1.8)
    
    # Arrow from Expert to Hedge
    add_straight_arrow(ax, x_between, y_expert - 0.35, x_between, y_hedge + 0.35,
                      color=colors['expert'], linewidth=1.8)
    
    # Arrow from Hedge to green forecast box on the right
    add_straight_arrow(ax, x_between + 0.6, y_hedge, x_right - 0.6, y_forecast,
                      color=colors['hedge'], linewidth=1.8)

# Add continuation arrows at edges
# Left edge - arrow coming from before
x_left_edge = time_positions[0]
x_before = 0
# Arrow to first Forecast box (cut off at edge)
add_straight_arrow(ax, x_before, y_hedge, time_positions[0] - 0.5, y_forecast,
                  color=colors['hedge'], linewidth=1.8)

# Right edge - arrow going to the next
x_right_edge = time_positions[-1]
x_after = 15.5
# Arrow from last Hedge box (cut off at edge)
# add_straight_arrow(ax, x_right_edge, y_hedge,
#                   x_after, y_forecast,
#                   color=colors['hedge'], linewidth=1.8)

# Add direct arrows from input to forecast for visual clarity (dashed, light)
# for x_pos in time_positions:
#     ax.plot([x_pos, x_pos], [y_input - 0.3, y_forecast + 0.3], 
#     color=colors['input'], linewidth=1, zorder=0)

# Add direct arrows from forecast to loss
for x_pos in time_positions:
    add_straight_arrow(ax, x_pos, y_input - 0.3, x_pos, y_forecast + 0.3,
                      color=colors['input'], linewidth=1.5)
    add_straight_arrow(ax, x_pos, y_forecast - 0.3, x_pos, y_loss + 0.3,
                      color=colors['output'], linewidth=1.5)

# Annotations
ax.text(16, y_input, 'CGM Data', ha='left', fontsize=9, 
        color=colors['input'], style='italic', weight='bold')
ax.text(16, y_forecast, 'Forecast', ha='left', fontsize=9, 
        color=colors['output'], style='italic', weight='bold')
ax.text(16, y_loss, 'Loss', ha='left', fontsize=9, 
        color=colors['loss'], style='italic', weight='bold')

# Bottom description
ax.text(8, 0.5, 
        'At each timestep: Experts use past losses ($\\ell_{t-1}$) and new data ($y_t$) to generate forecasts. ' +
        'Hedge weights predictions to produce $\\hat{y}_t$.',
        ha='center', fontsize=8.5, color=colors['text'],
        bbox=dict(boxstyle='round', facecolor=colors['light_bg'], 
                  alpha=0.9, edgecolor=colors['hedge'], linewidth=1.5))

# Bottom metrics
ax.text(8, 0.05, 
        '✓ 99.37% Clinical Reliability (Zone A+B)   ✓ 9.42 mg/dL RMSE @ 30-min   ✓ Fully Personalized', 
        ha='center', fontsize=8, color=colors['text'], weight='bold')

plt.tight_layout()

# Save as PDF
output_path = '/Users/abhishek/Documents/HedgeProject/overleaf/images/hbml_flowchart.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f'✓ Flowchart saved to: {output_path}')

# Also save as PNG for preview
png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
print(f'✓ PNG preview saved to: {png_path}')


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Load CSV with proper header handling
df = pd.read_csv("ablation_temperature.csv", delimiter="\t", header=[0, 1])

# Flatten multi-level column names and clean them
df.columns = [f"{col[0]}.{col[1]}" if col[1] != col[0] else col[0] for col in df.columns]

# Rename temperature column for easier access
df = df.rename(columns={"Temperature.Temperature": "Temperature"})

# Set up figure with paper-appropriate dimensions and high DPI
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

# Second y-axis for ASR
ax2 = ax1.twinx()

# Define methods and colors
methods = ["CoT Bypass", "Fake OverRefusal", "Reasoning Hijack"]
colors = ["tab:blue", "tab:orange", "tab:green"]

# Define markers for each method
markers = ["o", "s", "^"]  # circle, square, triangle

for method, color, marker in zip(methods, colors, markers):
    # Plot Harm (left axis, dashed)
    ax1.plot(df["Temperature"], df[f"{method}.Harm"],
             linestyle="--", color=color, linewidth=3.5, marker=marker,
             markersize=7, markerfacecolor=color, markeredgecolor=color,
             label=f"{method} Harm")
    # Plot ASR (right axis, solid)
    ax2.plot(df["Temperature"], df[f"{method}.ASR"],
             linestyle="-", color=color, linewidth=3.5, marker=marker,
             markersize=7, markerfacecolor=color, markeredgecolor=color,
             label=f"{method} ASR")

# Labels and styling
ax1.set_xlabel("Temperature", fontsize=28, fontfamily='Times New Roman', fontweight='bold')
ax1.set_ylabel("Harm Score", fontsize=28, fontfamily='Times New Roman', color='black', fontweight='bold')
ax2.set_ylabel("ASR (%)", fontsize=28, fontfamily='Times New Roman', color='black', fontweight='bold')

# Set tick label sizes
ax1.tick_params(axis='both', labelsize=23, labelcolor='black')
ax2.tick_params(axis='both', labelsize=23, labelcolor='black')

# Legends (combine both axes) - position outside plot area
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.5, 1.5),
#            fontsize=16, frameon=True, fancybox=True, shadow=False, prop=FontProperties(weight='bold', size=16))

# plt.title("Different Inference Temperatures", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("ablation_temperature.pdf", dpi=300)

plt.show()

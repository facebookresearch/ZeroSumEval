# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox
from zero_sum_eval.analysis.calculate_ratings import calculate_ratings
from utils import ROOT_DIR, ALL_DIRS, ROLE_WEIGHTS, CUSTOM_COLORS, LOGO_MAPPING, get_logo
# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Font settings for a more professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 16,  # Increased from 14
    'axes.titlesize': 18,  # Increased from 16
    'xtick.labelsize': 14,  # Increased from 12
    'ytick.labelsize': 14,  # Increased from 12
    'legend.fontsize': 14,  # Increased from 12
    'legend.title_fontsize': 16,  # Increased from 14
    'font.weight': 'bold'  # Make all text bold by default
})


ALL_DIRS = {
    "chess": "rankings-3-9-25_chess_predict_vs_cot",
    "mathquiz": "rankings-3-9-25_mathquiz_predict_vs_cot",
}

# Define model pairs (base model and its CoT variant)
MODEL_PAIRS = [
    ("gpt-4o", "gpt-4o-cot"),
    ("claude-3.7-sonnet", "claude-3.7-sonnet-cot"),
    ("gemini-2.0-flash", "gemini-2.0-flash-cot"),
    ("llama-3.1-70b", "llama-3.1-70b-cot"),
    ("llama-3.3-70b", "llama-3.3-70b-cot"),
    ("llama-3.1-405b", "llama-3.1-405b-cot"),
    ("qwen2.5-32b", "qwen2.5-32b-cot"),
    ("deepseek-chat", "deepseek-chat-cot"),
]

# Prepare data for plotting
game_ratings = {}
all_models = []

for game, dir in ALL_DIRS.items():
    if dir is None:
        continue

    # Extract all models for this comparison
    models_to_include = []
    for base, cot in MODEL_PAIRS:
        models_to_include.extend([base, cot])

    ratings = calculate_ratings(os.path.join(ROOT_DIR, dir),
                                bootstrap_rounds=100,
                                max_time_per_player=None,
                                models=models_to_include,
                                role_weights=ROLE_WEIGHTS[game])

    game_ratings[game] = ratings
    all_models.extend([model for model in ratings.index if model in models_to_include])

# Remove duplicates while preserving order
all_models = list(dict.fromkeys(all_models))


# Calculate global min and max for x-axis across all games and models
global_min_diff = float('inf')
global_max_diff = float('-inf')

for game in game_ratings:
    ratings = game_ratings[game]
    for base_model, cot_model in MODEL_PAIRS:
        base_rating = ratings['rating']['predicted'].get(base_model, 0)
        cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
        difference = cot_rating - base_rating
        global_min_diff = min(global_min_diff, difference)
        global_max_diff = max(global_max_diff, difference)

# Add some padding
x_padding = (global_max_diff - global_min_diff) * 0.1
global_min_diff -= x_padding
global_max_diff += x_padding

# Improved lollipop chart with better logo and value placement
def create_lollipop_chart():
    # Get all games
    games = list(game_ratings.keys())

    # Create figure with subplots - one per game
    fig, axes = plt.subplots(len(games), 1, figsize=(13, 3.2*len(games)), dpi=300)  # Reduced size for less whitespace
    if len(games) == 1:
        axes = [axes]

    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    for ax in axes:
        ax.set_facecolor('#FFFFFF')

    # Plot data for each game
    for i, game in enumerate(games):
        ax = axes[i]
        ratings = game_ratings[game]

        # Calculate differences for each model
        model_names = []
        differences = []
        colors = []

        for j, (base_model, cot_model) in enumerate(MODEL_PAIRS):
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)

            # Calculate difference (CoT - Predict)
            difference = cot_rating - base_rating
            differences.append(difference)

            # Add model name
            model_names.append(base_model)

            # Determine color based on difference
            colors.append(CUSTOM_COLORS[0] if difference > 0 else CUSTOM_COLORS[2])  # Bright blue vs Navy blue

        # Sort by difference value
        sorted_indices = np.argsort(differences)
        model_names = [model_names[idx] for idx in sorted_indices]
        differences = [differences[idx] for idx in sorted_indices]
        colors = [colors[idx] for idx in sorted_indices]

        # Calculate spacing for model names and logos
        max_name_length = max([len(name) for name in model_names])

        left_text_offset = -1 - (max_name_length * 0.01)  # For positive deltas
        right_text_offset = 1 + (max_name_length * 0.01)  # For negative deltas
        # Adjust value offset based on scale
        value_offset_pos = 5
        value_offset_neg = -5

        # Plot horizontal lines from zero to the difference
        for j, (model, diff, color) in enumerate(zip(model_names, differences, colors)):
            # Plot line
            ax.plot([0, diff], [j, j], color=color, linestyle='-', linewidth=4.0, alpha=0.8)  # Increased linewidth

            # Format model name
            formatted_name = model

            # Position model name based on whether delta is positive or negative
            if diff >= 0:  # Positive delta - name on left
                # Add model name on left
                ax.text(left_text_offset, j, formatted_name,
                       ha='right', va='center', fontsize=14, fontweight='bold')  # Increased size and made bold
            else:  # Negative delta - name on right
                # Add model name on right
                ax.text(right_text_offset, j, formatted_name,
                       ha='left', va='center', fontsize=14, fontweight='bold')  # Increased size and made bold

            # Add difference value near the logo
            if abs(diff) > 1:  # Only show non-zero differences
                # Position value based on direction
                value_offset = value_offset_pos if diff > 0 else value_offset_neg
                # Format value based on game (chess has larger numbers)
                value_text = f"{diff:.0f}"

                # Add text with white outline for better visibility
                text_obj = ax.text(diff + value_offset, j, value_text,
                       ha='left' if diff > 0 else 'right',
                       va='center', fontsize=12, color='black', fontweight='bold',  # Increased size
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

            # Add logo at the tip of the lollipop with increased size
            if model in LOGO_MAPPING:
                logo = get_logo(LOGO_MAPPING[model], size=0.22)  # Increased size
                if logo:
                    # Place logo at the tip of the lollipop
                    ab = AnnotationBbox(logo, (diff, j), xycoords='data',
                                      frameon=False, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)  # Increased linewidth

        # Set title for each game
        ax.set_title(f"{game.capitalize()}", fontsize=18, fontweight='bold')  # Increased size

        # Set x-axis label
        if i == len(games) - 1:  # Only add label to bottom subplot
            ax.set_xlabel('Rating Difference (CoT - Predict)', fontsize=16, fontweight='bold')  # Increased size and made bold

        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Set x-axis limits with global min and max
        ax.set_xlim(global_min_diff, global_max_diff)

        # Add annotations for interpretation
        ax.text(global_max_diff + 0.1, (len(model_names)-1)/2, 'CoT Better →',
               fontsize=14, ha='right', va='top', color=CUSTOM_COLORS[0], fontweight='bold')  # Increased size
        ax.text(global_min_diff - 0.1, (len(model_names)-1)/2, '← Predict Better',
               fontsize=14, ha='left', va='top', color=CUSTOM_COLORS[2], fontweight='bold')  # Increased size

        # Add subtle grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)

        # Clean up the frame
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')

    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray',
            ha='right', va='bottom', alpha=0.5, style='italic')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)  # Reduced spacing

    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_lollipop.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the improved lollipop chart
create_lollipop_chart()

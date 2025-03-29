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
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14
})

# Define Llama models to include in the comparison
LLAMA_MODELS = [
    "llama-3.3-70b",
    "llama-3.1-405b",
    "llama-3.1-70b",
    "llama-3.1-8b",
]

# Also include CoT variants if available
LLAMA_COT_MODELS = [
    "llama-3.3-70b-cot",
    "llama-3.1-405b-cot",
    "llama-3.1-70b-cot",
]

# Prepare data for plotting
game_ratings = {}
all_models = []

for game, dir in ALL_DIRS.items():
    if dir is None:
        continue
    
    # Extract all Llama models for this comparison
    models_to_include = LLAMA_MODELS + LLAMA_COT_MODELS
    
    ratings = calculate_ratings(os.path.join(ROOT_DIR, dir), 
                                bootstrap_rounds=100, 
                                max_time_per_player=None,
                                models=models_to_include,
                                role_weights=ROLE_WEIGHTS[game])
    
    game_ratings[game] = ratings
    all_models.extend([model for model in ratings.index if model in models_to_include])

# Remove duplicates while preserving order
all_models = list(dict.fromkeys(all_models))

# Function to create a grouped bar chart comparing Llama models across games
def create_grouped_bar_chart():
    # Get all games and models
    games = list(game_ratings.keys())
    models = [model for model in LLAMA_MODELS if model in all_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Number of models
    n_models = len(models)
    
    # Width of a bar 
    bar_width = 0.8 / n_models
    
    # Positions of the bars on the x-axis
    r = np.arange(len(games))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        # Get ratings for this model across all games
        model_ratings = []
        lower_errors = []
        upper_errors = []
        
        for game in games:
            if model in game_ratings[game].index:
                predicted = game_ratings[game]['rating']['predicted'].get(model, 0)
                lower = game_ratings[game]['rating']['lower'].get(model, 0)
                upper = game_ratings[game]['rating']['upper'].get(model, 0)
            else:
                predicted = 0
                lower = 0
                upper = 0
            
            model_ratings.append(predicted)
            lower_errors.append(max(0, predicted - lower))  # Ensure non-negative error
            upper_errors.append(max(0, upper - predicted))  # Ensure non-negative error
        
        # Calculate position for this model's bars
        pos = r + (i - n_models/2 + 0.5) * bar_width
        
        # Get color for this model
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        
        # Plot the bars
        bars = ax.bar(pos, model_ratings, width=bar_width, color=color, 
                     edgecolor='white', linewidth=0.5, label=model)
        
        # Add error bars
        ax.errorbar(pos, model_ratings, yerr=[lower_errors, upper_errors], 
                   fmt='none', ecolor='black', elinewidth=1, capsize=3, capthick=1, alpha=0.7)
        
        # Add model logo on top of each bar
        if model in LOGO_MAPPING:
            logo = get_logo(LOGO_MAPPING[model], size=0.15)
            if logo:
                for j, bar in enumerate(bars):
                    # Only add logo if the bar has a significant height
                    if model_ratings[j] > max(model_ratings) * 0.1:
                        ab = AnnotationBbox(logo, (pos[j], model_ratings[j]), 
                                          xybox=(0, 10), box_alignment=(0.5, 0),
                                          xycoords='data', boxcoords="offset points",
                                          frameon=False)
                        ax.add_artist(ab)
    
    # Add game names on the x-axis
    plt.xticks(r, [game.capitalize() for game in games], fontsize=14, fontweight='bold')
    
    # Add a legend
    legend = ax.legend(loc='upper right', frameon=True, fontsize=12, 
                      title="Llama Models", title_fontsize=14)
    
    # Set labels and title
    ax.set_ylabel('Rating', fontsize=16, labelpad=15)
    ax.set_title('Llama Model Performance Comparison', fontsize=20, fontweight='bold', pad=20)
    
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Clean up the frame
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray', 
            ha='right', va='bottom', alpha=0.5, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('paper/figures/llama_bar_comparison.pdf', dpi=300, bbox_inches='tight')

# Function to create a comparison between base models and their CoT variants
def create_cot_comparison():
    # Define model pairs (base model and its CoT variant)
    MODEL_PAIRS = [
        ("llama-3.3-70b", "llama-3.3-70b-cot"),
        ("llama-3.1-405b", "llama-3.1-405b-cot"),
        ("llama-3.1-70b", "llama-3.1-70b-cot"),
    ]
    COT_DIRS = {
        "chess": "rankings-3-9-25_chess_predict_vs_cot",
        "mathquiz": "rankings-3-9-25_mathquiz_predict_vs_cot",
    }
    game_ratings = {}
    for game, dir in COT_DIRS.items():
        if dir is None:
            continue
        
        # Extract all Llama models for this comparison
        models_to_include = LLAMA_MODELS + LLAMA_COT_MODELS
        
        ratings = calculate_ratings(os.path.join(ROOT_DIR, dir), 
                                    bootstrap_rounds=100, 
                                    max_time_per_player=None,
                                    models=models_to_include,
                                    role_weights=ROLE_WEIGHTS[game])
        
        game_ratings[game] = ratings

    # Get all games
    games = list(game_ratings.keys())
    
    # Calculate global min and max for x-axis across all games and models
    global_min_diff = float('inf')
    global_max_diff = float('-inf')
    
    for game in games:
        ratings = game_ratings[game]
        for base_model, cot_model in MODEL_PAIRS:
            if base_model in ratings.index and cot_model in ratings.index:
                base_rating = ratings['rating']['predicted'].get(base_model, 0)
                cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
                difference = cot_rating - base_rating
                global_min_diff = min(global_min_diff, difference)
                global_max_diff = max(global_max_diff, difference)
    
    # Add some padding
    x_padding = (global_max_diff - global_min_diff) * 0.1
    global_min_diff -= x_padding
    global_max_diff += x_padding
    
    # Create figure with subplots - one per game
    fig, axes = plt.subplots(len(games), 1, figsize=(14, 3.5*len(games)), dpi=300)
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
            # Skip if either model is not in the ratings
            if base_model not in ratings.index or cot_model not in ratings.index:
                continue
                
            # Get ratings
            base_rating = ratings['rating']['predicted'].get(base_model, 0)
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            
            # Calculate difference (CoT - Base)
            difference = cot_rating - base_rating
            differences.append(difference)
            
            # Add model name
            model_names.append(base_model)
            
            # Determine color based on difference
            colors.append(CUSTOM_COLORS[0] if difference > 0 else CUSTOM_COLORS[2])  # Bright blue vs Navy blue
        
        # Skip if no valid model pairs for this game
        if not model_names:
            ax.text(0.5, 0.5, f"No data for {game.capitalize()}", 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f"{game.capitalize()}", fontsize=16, fontweight='bold')
            continue
        
        # Sort by difference value
        sorted_indices = np.argsort(differences)
        model_names = [model_names[idx] for idx in sorted_indices]
        differences = [differences[idx] for idx in sorted_indices]
        colors = [colors[idx] for idx in sorted_indices]
        
        # Calculate spacing for model names and logos
        max_name_length = max([len(name) for name in model_names])
        left_text_offset = -1 - (max_name_length * 0.01)  # For positive deltas
        right_text_offset = 1 + (max_name_length * 0.01)  # For negative deltas
        value_offset_pos = 5
        value_offset_neg = -5
        
        # Plot horizontal lines from zero to the difference
        for j, (model, diff, color) in enumerate(zip(model_names, differences, colors)):
            # Plot line
            ax.plot([0, diff], [j, j], color=color, linestyle='-', linewidth=3.5, alpha=0.8)
            
            # Format model name
            formatted_name = model
            
            # Position model name based on whether delta is positive or negative
            if diff >= 0:  # Positive delta - name on left
                # Add model name on left
                ax.text(left_text_offset, j, formatted_name, 
                       ha='right', va='center', fontsize=12, fontweight='medium')
            else:  # Negative delta - name on right
                # Add model name on right
                ax.text(right_text_offset, j, formatted_name, 
                       ha='left', va='center', fontsize=12, fontweight='medium')
            
            # Add difference value near the logo
            if abs(diff) > 1:  # Only show non-zero differences
                # Position value based on direction
                value_offset = value_offset_pos if diff > 0 else value_offset_neg
                # Format value based on game (chess has larger numbers)
                value_text = f"{diff:.0f}"
                
                # Add text with white outline for better visibility
                text_obj = ax.text(diff + value_offset, j, value_text,
                       ha='left' if diff > 0 else 'right',
                       va='center', fontsize=10, color='black', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # Add logo at the tip of the lollipop with increased size
            if model in LOGO_MAPPING:
                logo = get_logo(LOGO_MAPPING[model], size=0.18)
                if logo:
                    # Place logo at the tip of the lollipop
                    ab = AnnotationBbox(logo, (diff, j), xycoords='data',
                                      frameon=False, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Set title for each game
        ax.set_title(f"{game.capitalize()}", fontsize=16, fontweight='bold')
        
        # Set x-axis label
        if i == len(games) - 1:  # Only add label to bottom subplot
            ax.set_xlabel('Rating Difference (CoT - Base)', fontsize=12)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis limits with global min and max
        ax.set_xlim(global_min_diff, global_max_diff)
        
        # Add annotations for interpretation
        ax.text(global_max_diff + 0.1, len(model_names)-1, 'CoT Better →', 
               fontsize=12, ha='right', va='top', color=CUSTOM_COLORS[0], fontweight='bold')
        ax.text(global_min_diff - 0.1, len(model_names)-1, '← Base Better', 
               fontsize=12, ha='left', va='top', color=CUSTOM_COLORS[2], fontweight='bold')
        
        # Add subtle grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Clean up the frame
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
    
    # Add a main title
    fig.suptitle('Llama: Chain-of-Thought vs Base Model Performance', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray', 
            ha='right', va='bottom', alpha=0.5, style='italic')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)
    
    # Save the figure
    plt.savefig('paper/figures/llama_cot_comparison.pdf', dpi=300, bbox_inches='tight')

# Call the visualization functions
create_grouped_bar_chart()
# create_cot_comparison()

import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox
from zero_sum_eval.analysis.calculate_ratings import calculate_ratings
from utils import ROOT_DIR, ALL_DIRS, ROLE_WEIGHTS, CUSTOM_COLORS, GAME_COLOR_MAPPING, LOGO_MAPPING, get_logo
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


# Define model pairs (base model and its CoT variant)
MODEL_PAIRS = [
    ("gpt-4o", "o3-mini-high"),
    ("claude-3.7-sonnet", "claude-3.7-sonnet-thinking"),
    ("qwen2.5-32b", "qwq-32b"),
    ("deepseek-chat", "deepseek-r1"),
]

# Prepare data for plotting
game_ratings = {}
all_models = []

for game, dir in ALL_DIRS.items():
    if dir is None:
        continue
    
    # Extract all models for this comparison
    models_to_include = []
    for cot, thinking in MODEL_PAIRS:
        models_to_include.extend([cot, thinking])
    
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
    for cot_model, thinking_model in MODEL_PAIRS:
        cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
        thinking_rating = ratings['rating']['predicted'].get(thinking_model, 0)
        difference = thinking_rating - cot_rating
        global_min_diff = min(global_min_diff, difference)
        global_max_diff = max(global_max_diff, difference)

# Add some padding
x_padding = (global_max_diff - global_min_diff) * 0.1
global_min_diff -= x_padding
global_max_diff += x_padding

# Create a grouped lollipop chart with all games in one plot
def create_grouped_lollipop_chart():
    # Get all games
    games = list(game_ratings.keys())
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)  # Reduced size for less whitespace
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Define spacing parameters
    model_height = 1.2  # Slightly reduced height allocated for each model
    game_spacing = 0.25  # Reduced spacing between games within a model group
    group_spacing = 0.8  # Reduced spacing between model groups
    
    # Calculate total number of model pairs
    num_model_pairs = len(MODEL_PAIRS)
    
    # Track y-positions for labels
    model_y_positions = {}
    
    # Track min and max differences for x-axis limits
    all_differences = []
    
    # Plot data for each model pair
    for model_idx, (cot_model, thinking_model) in enumerate(MODEL_PAIRS):
        # Base y-position for this model pair
        base_y = (num_model_pairs - model_idx - 1) * (model_height + group_spacing)
        
        # Store the y-position for this model
        model_y_positions[thinking_model] = base_y + (len(games) * game_spacing) / 2
        
        # Plot data for each game within this model pair
        game_differences = []
        
        for game_idx, game in enumerate(games):
            # Calculate y-position for this game
            y_pos = base_y + game_idx * game_spacing
            
            # Get ratings
            ratings = game_ratings[game]
            cot_rating = ratings['rating']['predicted'].get(cot_model, 0)
            thinking_rating = ratings['rating']['predicted'].get(thinking_model, 0)
            
            # Calculate difference (Thinking - CoT)
            difference = thinking_rating - cot_rating
            all_differences.append(difference)
            game_differences.append(difference)
            
            # Get color from the game color mapping
            game_color = GAME_COLOR_MAPPING.get(game, CUSTOM_COLORS[game_idx % len(CUSTOM_COLORS)])
            
            # Plot horizontal line from zero to the difference
            ax.plot([0, difference], [y_pos, y_pos], color=game_color, linestyle='-', linewidth=4.0, alpha=0.8)  # Increased linewidth
            
            # Add a marker at the end
            ax.scatter(difference, y_pos, color=game_color, s=120, alpha=0.9, zorder=3)  # Increased marker size
            
            # Add game label next to the y-axis
            game_name = game.capitalize()
            
            # Place game name based on difference value
            if difference < 0:
                # Place game name on the right side of y-axis
                ax.text(20, y_pos, game_name, 
                       ha='left', va='center', fontsize=12, fontweight='bold',  # Increased size and made bold
                       color='#333333')  # Darker color for better readability
            else:
                # Place game name on the left side of y-axis
                ax.text(-20, y_pos, game_name, 
                       ha='right', va='center', fontsize=12, fontweight='bold',  # Increased size and made bold
                       color='#333333')  # Darker color for better readability
            
            # Add difference value at the tip of the lollipop
            if abs(difference) > 1:  # Only show non-zero differences
                value_offset = 20
                value_text = f"{difference:.0f}"
                
                # Position the value text at the tip
                if difference < 0:  # CoT better
                    ax.text(difference - value_offset, y_pos, value_text, 
                           ha='right', va='center', fontsize=11, color='black', fontweight='bold',  # Increased size
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                else:  # Thinking better
                    ax.text(difference + value_offset, y_pos, value_text, 
                           ha='left', va='center', fontsize=11, color='black', fontweight='bold',  # Increased size
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add model logo instead of name on the right side
        if thinking_model in LOGO_MAPPING:
            # Get the logo
            logo = get_logo(LOGO_MAPPING[thinking_model], size=0.8)  # Increased logo size
            if logo:
                # Calculate position for logo - ensure it's to the right of all lollipops
                logo_x = 500 # Position well to the right
                logo_y = model_y_positions[thinking_model]-.1
                # Add the logo
                ab = AnnotationBbox(logo, (logo_x, logo_y), 
                                  xycoords='data', frameon=False, 
                                  box_alignment=(0.5, 0.5), zorder=10)
                ax.add_artist(ab)
        
        # Add a horizontal line to separate model groups (except after the last one)
        if model_idx < num_model_pairs - 1:
            separator_y = base_y - group_spacing / 2
            ax.axhline(y=separator_y, color='#dddddd', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)  # Increased linewidth
    
    # Set x-axis label
    ax.set_xlabel('Rating Difference (Thinking - CoT)', fontsize=16, fontweight='bold', labelpad=10)  # Increased size and made bold
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add padding to x-axis limits and extend right side for logos
    x_padding = (max(all_differences) - min(all_differences)) * 0.15
    # Extend right side more to make room for logos
    # ax.set_xlim(min(all_differences) - x_padding, abs(max(all_differences)) * 1.4)
    ax.set_xlim(-1100, 550)
    # Add annotations for interpretation - using blue shades instead of green/red
    positive_color = CUSTOM_COLORS[0]  # Bright blue for positive differences
    negative_color = CUSTOM_COLORS[2]  # Navy blue for negative differences
    
    ax.text(max(all_differences) + x_padding * 0.9, num_model_pairs * (model_height + group_spacing) - 0.5, 
           'Thinking Better →', fontsize=14, ha='right', va='top',  # Increased size
           color=positive_color, fontweight='bold')
    ax.text(min(all_differences) - x_padding * 0.9, num_model_pairs * (model_height + group_spacing) - 0.5, 
           '← CoT Better', fontsize=14, ha='left', va='top',  # Increased size
           color=negative_color, fontweight='bold')
    
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
    
    # Update the file paths for saving the figures
    plt.savefig('paper/figures/reasoning_vs_cot_grouped.pdf', dpi=300, bbox_inches='tight')

# Call the function to create the grouped lollipop chart
create_grouped_lollipop_chart()
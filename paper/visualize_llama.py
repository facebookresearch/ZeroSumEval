import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from zero_sum_eval.analysis.calculate_ratings import calculate_ratings

# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - using different shades of blue
CUSTOM_COLORS = [
    (14/255, 140/255, 247/255),   # Bright blue
    (41/255, 44/255, 147/255),    # Deep blue
    (0/255, 84/255, 159/255),     # Navy blue
    (86/255, 180/255, 233/255),   # Sky blue
    (120/255, 180/255, 210/255),  # Darker light blue for mathquiz
    (0/255, 119/255, 182/255),    # Medium blue
    (65/255, 105/255, 225/255)    # Royal blue
]

# Create a mapping between games and their colors for consistency
GAME_COLOR_MAPPING = {
    "chess": CUSTOM_COLORS[0],
    "debate": CUSTOM_COLORS[1],
    "gandalf": CUSTOM_COLORS[2],
    "liars_dice": CUSTOM_COLORS[3],
    "mathquiz": CUSTOM_COLORS[4],
    "poker": CUSTOM_COLORS[5],
    "pyjail": CUSTOM_COLORS[6]
}

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

# Function to load and resize logo with circular cropping and guaranteed white background
def get_logo(logo_path, size=0.15):
    try:
        # Use PIL directly to load the image
        pil_img = Image.open(logo_path)
        
        # Force conversion to RGBA first to properly handle all image types
        if pil_img.mode == 'P':  # Palette mode
            pil_img = pil_img.convert('RGBA')
        elif pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        
        # Create a square image by cropping
        width, height = pil_img.size
        size_px = min(width, height)
        
        # Calculate crop box (centered)
        left = (width - size_px) // 2
        top = (height - size_px) // 2
        right = left + size_px
        bottom = top + size_px
        
        # Crop to square
        square_img = pil_img.crop((left, top, right, bottom))
        
        # Create a solid white background image
        white_bg = Image.new('RGB', (size_px, size_px), (255, 255, 255))
        
        # Create a circular mask
        mask = Image.new('L', (size_px, size_px), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size_px, size_px), fill=255)
        
        # Paste the logo onto the white background using the mask
        # This is the key step - we're using the RGBA image as the source but pasting onto RGB
        white_bg.paste(square_img, (0, 0), square_img.split()[3])  # Use alpha channel as mask
        
        # Add a border
        draw_border = ImageDraw.Draw(white_bg)
        draw_border.ellipse((0, 0, size_px-1, size_px-1), outline=(50, 50, 50), width=2)
        
        # Resize to a standard size
        target_size = (100, 100)
        resized_img = white_bg.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array for matplotlib
        img_array = np.array(resized_img)
        
        # Create an OffsetImage with the standardized image
        offset_img = OffsetImage(img_array, zoom=size)
        
        # Force the image to have a white background in matplotlib
        offset_img.set_zorder(10)  # Ensure it's on top
        
        return offset_img
    except Exception as e:
        print(f"Error loading logo {logo_path}: {e}")
        # Create a fallback simple circle with the first letter of the model
        model_name = os.path.basename(logo_path).split('.')[0]
        first_letter = model_name[0].upper() if model_name else "?"
        
        # Create a white circle with text
        fallback = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(fallback)
        draw.ellipse((0, 0, 99, 99), outline=(100, 100, 100), width=2)
        
        # Add text (centered)
        try:
            # Try to use a font if available
            from PIL import ImageFont
            font = ImageFont.truetype("Arial", 40)
            text_width, text_height = draw.textsize(first_letter, font=font)
            draw.text((50-text_width//2, 50-text_height//2), first_letter, fill=(0, 0, 0), font=font)
        except:
            # Fallback if font not available
            draw.text((40, 30), first_letter, fill=(0, 0, 0))
        
        img_array = np.array(fallback)
        return OffsetImage(img_array, zoom=size)

# Map model names to their logo files
LOGO_DIR = "paper/logos"
LOGO_MAPPING = {
    "llama-3.3-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-8b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b-cot": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b-cot": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.3-70b-cot": os.path.join(LOGO_DIR, "llama.png"),
}

ROOT_DIR = "/Users/haidark/Library/CloudStorage/GoogleDrive-haidark@gmail.com/My Drive/Zero Sum Eval/rankings-3-9-25/"
ALL_DIRS = {
    "chess": "rankings-3-9-25_chess",
    "debate": "rankings-3-9-25_debate",
    "gandalf": "rankings-3-9-25_gandalf_final_500",
    "liars_dice": "rankings-3-9-25_liars_dice_reasoning_1000",
    "mathquiz": "rankings-3-9-25_mathquiz_final_500",
    "poker": "rankings-3-9-25_poker_final_500",
    "pyjail": None  # "rankings-3-9-25_pyjail"
}

ROLE_WEIGHTS = {
    "chess": {
        "white": 1.0,
        "black": 2.0
    },
    "debate": None,
    "gandalf": {
        "sentinel": 1.0,
        "infiltrator": 2.0
    },
    "liars_dice": None,
    "mathquiz": {
        "student": 1.0,
        "teacher": 2.0
    },
    "poker": None,
    "pyjail": {
        "defender": 2.0,    
        "attacker": 1.0
    }
}

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

# Function to create a radar chart comparing Llama models across games
def create_radar_chart():
    # Get all games and models
    games = list(game_ratings.keys())
    models = [model for model in LLAMA_MODELS if model in all_models]
    
    # Number of variables
    N = len(games)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [game.capitalize() for game in games], fontsize=14, fontweight='bold')
    
    # Draw ylabels (rating values)
    ax.set_rlabel_position(0)
    
    # Find max rating across all games and models for scaling
    max_rating = 0
    for game in games:
        for model in models:
            if model in game_ratings[game].index:
                rating = game_ratings[game]['rating']['predicted'].get(model, 0)
                max_rating = max(max_rating, rating)
    
    # Add some padding to the max rating
    max_rating = max_rating * 1.1
    plt.ylim(0, max_rating)
    
    # Plot each model
    for i, model in enumerate(models):
        # Get ratings for this model across all games
        model_ratings = []
        for game in games:
            if model in game_ratings[game].index:
                rating = game_ratings[game]['rating']['predicted'].get(model, 0)
            else:
                rating = 0
            model_ratings.append(rating)
        
        # Close the loop
        model_ratings += model_ratings[:1]
        
        # Get color for this model - use a distinct color from the palette
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        
        # Plot the ratings
        ax.plot(angles, model_ratings, linewidth=3, linestyle='solid', color=color, alpha=0.8, label=model)
        ax.fill(angles, model_ratings, color=color, alpha=0.1)
    
    # Add legend with model logos
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, 
                      fontsize=12, title="Llama Models", title_fontsize=14)
    
    # Set title
    plt.title('Llama Model Performance Across Games', fontsize=20, fontweight='bold', pad=20)
    
    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray', 
            ha='right', va='bottom', alpha=0.5, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('paper/figures/llama_radar_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/llama_radar_comparison.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('paper/figures/llama_bar_comparison.png', dpi=300, bbox_inches='tight')

# Function to create a heatmap comparing Llama models across games
def create_heatmap():
    # Get all games and models
    games = list(game_ratings.keys())
    models = [model for model in LLAMA_MODELS + LLAMA_COT_MODELS if model in all_models]
    
    # Create a matrix of ratings
    ratings_matrix = np.zeros((len(models), len(games)))
    
    # Fill the matrix with ratings
    for i, model in enumerate(models):
        for j, game in enumerate(games):
            if model in game_ratings[game].index:
                ratings_matrix[i, j] = game_ratings[game]['rating']['predicted'].get(model, 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Create the heatmap
    im = ax.imshow(ratings_matrix, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Rating', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(games)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([game.capitalize() for game in games], fontsize=12)
    ax.set_yticklabels(models, fontsize=12)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in each cell
    for i in range(len(models)):
        for j in range(len(games)):
            text = ax.text(j, i, f"{ratings_matrix[i, j]:.1f}",
                          ha="center", va="center", color="white" if ratings_matrix[i, j] > np.max(ratings_matrix)/2 else "black",
                          fontsize=10, fontweight='bold')
    
    # Set title
    ax.set_title('Llama Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Add a subtle watermark
    fig.text(0.98, 0.02, 'ZeroSumEval', fontsize=8, color='gray', 
            ha='right', va='bottom', alpha=0.5, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('paper/figures/llama_heatmap_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/llama_heatmap_comparison.png', dpi=300, bbox_inches='tight')

# Function to create a comparison between base models and their CoT variants
def create_cot_comparison():
    # Define model pairs (base model and its CoT variant)
    MODEL_PAIRS = [
        ("llama-3.3-70b", "llama-3.3-70b-cot"),
        ("llama-3.1-405b", "llama-3.1-405b-cot"),
        ("llama-3.1-70b", "llama-3.1-70b-cot"),
    ]
    
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
                       ha='right', va='center', fontsize=11, fontweight='medium')
            else:  # Negative delta - name on right
                # Add model name on right
                ax.text(right_text_offset, j, formatted_name, 
                       ha='left', va='center', fontsize=11, fontweight='medium')
            
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
               fontsize=11, ha='right', va='top', color=CUSTOM_COLORS[0], fontweight='bold')
        ax.text(global_min_diff - 0.1, len(model_names)-1, '← Base Better', 
               fontsize=11, ha='left', va='top', color=CUSTOM_COLORS[2], fontweight='bold')
        
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
    plt.savefig('paper/figures/llama_cot_comparison.png', dpi=300, bbox_inches='tight')

# Call the visualization functions
create_radar_chart()
create_grouped_bar_chart()
create_heatmap()
create_cot_comparison()

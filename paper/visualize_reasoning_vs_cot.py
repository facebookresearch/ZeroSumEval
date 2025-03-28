import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from zero_sum_eval.analysis.calculate_ratings import calculate_ratings

# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - more visually appealing
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
    "gpt-4o": os.path.join(LOGO_DIR, "gpt-4.png"),
    "claude-3.7-sonnet": os.path.join(LOGO_DIR, "claude.png"),
    "claude-3.7-sonnet-thinking": os.path.join(LOGO_DIR, "claude.png"),
    "gemini-2.0-flash": os.path.join(LOGO_DIR, "gemini.png"),
    "llama-3.3-70b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-405b": os.path.join(LOGO_DIR, "llama.png"),
    "llama-3.1-70b": os.path.join(LOGO_DIR, "llama.png"),
    "deepseek-chat": os.path.join(LOGO_DIR, "deepseek.png"),
    "deepseek-r1": os.path.join(LOGO_DIR, "deepseek.png"),
    "qwen2.5-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "qwq-32b": os.path.join(LOGO_DIR, "qwen2.png"),
    "o3-mini-high": os.path.join(LOGO_DIR, "openai.png")
}

ROOT_DIR = "/Users/haidark/Library/CloudStorage/GoogleDrive-haidark@gmail.com/My Drive/Zero Sum Eval/rankings-3-9-25/"
ALL_DIRS = {
    "chess": "rankings-3-9-25_chess",
    "gandalf": "rankings-3-9-25_gandalf_final_500",
    "liars_dice": "rankings-3-9-25_liars_dice_reasoning_1000",
    "mathquiz": "rankings-3-9-25_mathquiz_final_500",
    "poker": "rankings-3-9-25_poker_final_500",
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
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Set background color
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Define spacing parameters
    model_height = 1.0  # Height allocated for each model
    game_spacing = 0.3  # Spacing between games within a model group
    group_spacing = 1.0  # Spacing between model groups
    
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
            ax.plot([0, difference], [y_pos, y_pos], color=game_color, linestyle='-', linewidth=3.5, alpha=0.8)
            
            # Add a marker at the end
            ax.scatter(difference, y_pos, color=game_color, s=100, alpha=0.9, zorder=3)
            
            # Add game label next to the y-axis
            game_name = game.capitalize()
            
            # Place game name based on difference value
            if difference < 0:
                # Place game name on the right side of y-axis
                ax.text(20, y_pos, game_name, 
                       ha='left', va='center', fontsize=10, fontweight='medium',
                       color='#555555')
            else:
                # Place game name on the left side of y-axis
                ax.text(-20, y_pos, game_name, 
                       ha='right', va='center', fontsize=10, fontweight='medium',
                       color='#555555')
            
            # Add difference value at the tip of the lollipop
            if abs(difference) > 1:  # Only show non-zero differences
                value_offset = 20
                value_text = f"{difference:.0f}"
                
                # Position the value text at the tip
                if difference < 0:  # CoT better
                    ax.text(difference - value_offset, y_pos, value_text, 
                           ha='right', va='center', fontsize=9, color='black', fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                else:  # Thinking better
                    ax.text(difference + value_offset, y_pos, value_text, 
                           ha='left', va='center', fontsize=9, color='black', fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add model logo instead of name on the right side
        if thinking_model in LOGO_MAPPING:
            # Get the logo
            logo = get_logo(LOGO_MAPPING[thinking_model], size=0.7)
            if logo:
                # Calculate position for logo - ensure it's to the right of all lollipops
                logo_x = 1050 # Position well to the right
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
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Set title
    ax.set_title('Rating Difference: Thinking vs. Chain-of-Thought', fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis label
    ax.set_xlabel('Rating Difference (Thinking - CoT)', fontsize=12, labelpad=10)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add padding to x-axis limits and extend right side for logos
    x_padding = (max(all_differences) - min(all_differences)) * 0.15
    # Extend right side more to make room for logos
    ax.set_xlim(min(all_differences) - x_padding, max(all_differences) * 1.4)
    
    # Add annotations for interpretation - using blue shades instead of green/red
    positive_color = CUSTOM_COLORS[0]  # Bright blue for positive differences
    negative_color = CUSTOM_COLORS[2]  # Navy blue for negative differences
    
    ax.text(max(all_differences) + x_padding * 0.9, num_model_pairs * (model_height + group_spacing) - 0.5, 
           'Thinking Better →', fontsize=11, ha='right', va='top', 
           color=positive_color, fontweight='bold')
    ax.text(min(all_differences) - x_padding * 0.9, num_model_pairs * (model_height + group_spacing) - 0.5, 
           '← CoT Better', fontsize=11, ha='left', va='top', 
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
    plt.savefig('paper/figures/reasoning_vs_cot_grouped.png', dpi=300, bbox_inches='tight')

# Call the function to create the grouped lollipop chart
create_grouped_lollipop_chart()
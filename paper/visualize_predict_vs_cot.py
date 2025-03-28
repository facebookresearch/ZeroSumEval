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
    "chess": "rankings-3-9-25_chess_predict_vs_cot",
    "mathquiz": "rankings-3-9-25_mathquiz_predict_vs_cot",
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
            ax.set_xlabel('Rating Difference (CoT - Predict)', fontsize=12)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Set x-axis limits with global min and max
        ax.set_xlim(global_min_diff, global_max_diff)
        
        # Add annotations for interpretation
        ax.text(global_max_diff + 0.1, len(model_names)-1, 'CoT Better →', 
               fontsize=11, ha='right', va='top', color=CUSTOM_COLORS[0], fontweight='bold')
        ax.text(global_min_diff - 0.1, len(model_names)-1, '← Predict Better', 
               fontsize=11, ha='left', va='top', color=CUSTOM_COLORS[2], fontweight='bold')
        
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
    plt.subplots_adjust(top=0.92, hspace=0.4)
    
    # Save the figure
    plt.savefig('paper/figures/cot_vs_predict_lollipop.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/cot_vs_predict_lollipop.png', dpi=300, bbox_inches='tight')

# Call the function to create the improved lollipop chart
create_lollipop_chart()
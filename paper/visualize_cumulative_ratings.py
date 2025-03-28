import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
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

# Function to load and resize logo
def get_logo(logo_path, size=0.15):
    try:
        # Use PIL directly to load the image
        pil_img = Image.open(logo_path)
        
        # Convert to RGBA if needed
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        
        # Resize all logos to a standard size (e.g., 100x100 pixels)
        target_size = (100, 100)
        
        # Calculate aspect ratio
        aspect = pil_img.width / pil_img.height
        
        # Resize maintaining aspect ratio
        if aspect > 1:  # Width > Height
            new_width = target_size[0]
            new_height = int(new_width / aspect)
        else:  # Height >= Width
            new_height = target_size[1]
            new_width = int(new_height * aspect)
        
        # Create a new transparent image with the target size
        new_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
        
        # Resize the original image
        resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position to paste (center)
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        
        # Paste the resized image onto the transparent canvas
        new_img.paste(resized_img, (paste_x, paste_y), resized_img if resized_img.mode == 'RGBA' else None)
        
        # Convert to numpy array for matplotlib
        img_array = np.array(new_img)
        
        # Create an OffsetImage with the standardized image
        return OffsetImage(img_array, zoom=size)
    except Exception as e:
        print(f"Error loading logo {logo_path}: {e}")
        return None

# Map model names to their logo files
LOGO_DIR = "paper/logos"
LOGO_MAPPING = {
    "gpt-4o": os.path.join(LOGO_DIR, "openai.png"),
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

# Prepare data for a single plot
all_players = set()
game_ratings = {}

for game, dir in ALL_DIRS.items():
    if dir is None:
        continue
    ratings = calculate_ratings(os.path.join(ROOT_DIR, dir), 
                                bootstrap_rounds=100, 
                                max_time_per_player=None, 
                                role_weights=ROLE_WEIGHTS[game])
    
    game_ratings[game] = ratings
    all_players.update(ratings.index)

# Create a list of all players
all_players = list(all_players)

# Calculate total ratings for each player across all games
total_ratings = {}
for player in all_players:
    total_ratings[player] = sum(game_ratings[game]['rating']['predicted'].get(player, 0) for game in game_ratings)

# Sort players by total ratings (highest to lowest)
sorted_players = sorted(all_players, key=lambda x: total_ratings[x], reverse=True)

# Initialize data structures for plotting
num_games = len(game_ratings)
index = np.arange(len(sorted_players))

# Create a figure with a specific aspect ratio
fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

# Set background color
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Initialize arrays for the bottom of the bars and cumulative errors
cumulative_ratings = np.zeros(len(sorted_players))
cumulative_lower_errors = np.zeros(len(sorted_players))
cumulative_upper_errors = np.zeros(len(sorted_players))

# Plot each game's ratings as a stacked bar
for i, (game, ratings) in enumerate(game_ratings.items()):
    # Get ratings and error bounds for each player
    game_ratings_values = []
    lower_bounds = []
    upper_bounds = []
    
    for player in sorted_players:
        predicted = ratings['rating']['predicted'].get(player, 0)
        lower = ratings['rating']['lower'].get(player, 0)
        upper = ratings['rating']['upper'].get(player, 0)
        
        game_ratings_values.append(predicted)
        lower_bounds.append(max(0, predicted - lower))  # Ensure non-negative error
        upper_bounds.append(max(0, upper - predicted))  # Ensure non-negative error
    
    # Plot the bars with a slight gap between them - use specific color for each game
    game_color = GAME_COLOR_MAPPING.get(game, CUSTOM_COLORS[i % len(CUSTOM_COLORS)])
    bars = ax.bar(index, game_ratings_values, width=0.8, label=game.capitalize(), 
           color=game_color, bottom=cumulative_ratings, 
           edgecolor='white', linewidth=0.5)
    
    # Add error bars for this game segment
    ax.errorbar(
        index, 
        cumulative_ratings + np.array(game_ratings_values), 
        yerr=[lower_bounds, upper_bounds],
        fmt='none',  # No connecting line
        ecolor='black',
        elinewidth=1.2,
        capsize=4,
        capthick=1.5,
        alpha=0.7,
        zorder=10  # Ensure error bars are drawn on top
    )
    
    # Add the individual game rating values in the middle of each bar segment
    for j, bar in enumerate(bars):
        if game_ratings_values[j] > 0.5:  # Only add text if the bar segment is tall enough
            # Calculate the middle position of this bar segment
            bar_height = game_ratings_values[j]
            y_pos = cumulative_ratings[j] + (bar_height / 2)
            
            # Add the game name only for the first and last bars
            if j == 0:
                ax.annotate(f'{game.capitalize()}', 
                            xy=(j, y_pos),
                            xytext=(0, 5),  # Position above the rating
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=12, fontweight='bold',
                            color='white')
            
            # Add the rating value for all bars
            ax.annotate(f'{game_ratings_values[j]:.1f}', 
                        xy=(j, y_pos),
                        xytext=(0, -10),  # Position below the game name
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='white')
    
    # Update cumulative values for the next game
    cumulative_ratings += np.array(game_ratings_values)
    cumulative_lower_errors += np.array(lower_bounds)
    cumulative_upper_errors += np.array(upper_bounds)

# Add logos on top of each bar and total values
for i, player in enumerate(sorted_players):
    if player in LOGO_MAPPING:
        logo_path = LOGO_MAPPING[player]
        logo_image = get_logo(logo_path, size=0.5)  # Consistent size for all logos
        if logo_image:
            # Position the logo at the top of the bar
            ab = AnnotationBbox(
                logo_image, 
                (i, cumulative_ratings[i]), 
                xybox=(0, 40),  # Offset above the bar
                xycoords='data',
                boxcoords="offset points",
                frameon=False
            )
            ax.add_artist(ab)
    

# Add a horizontal line at y=0
ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.3)

# Customize the grid
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
ax.set_axisbelow(True)  # Put grid behind bars

# Set title and labels with enhanced typography
ax.set_title('Cumulative Ratings', fontsize=20, fontweight='bold', pad=20)
# ax.set_xlabel('Model', fontsize=16, labelpad=15)
ax.set_ylabel('Rating', fontsize=16, labelpad=15)

# Format x-axis labels
plt.xticks(index, [p.replace('-', '\n') for p in sorted_players], rotation=0, ha='center', fontsize=12)

# Format y-axis with fewer ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.yticks(fontsize=12)

# Add a subtle box around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#DDDDDD')
    spine.set_linewidth(0.5)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend

# Save the figure with high resolution
plt.savefig('paper/figures/model_performance_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')

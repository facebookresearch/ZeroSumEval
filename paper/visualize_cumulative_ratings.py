# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
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
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)  # Reduced size for less whitespace

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
        elinewidth=1.5,  # Increased linewidth
        capsize=5,  # Increased capsize
        capthick=1.8,  # Increased thickness
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
                            fontsize=14, fontweight='bold',  # Increased size
                            color='white')

            # Add the rating value for all bars
            ax.annotate(f'{game_ratings_values[j]:.1f}',
                        xy=(j, y_pos),
                        xytext=(0, -10),  # Position below the game name
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',  # Increased size
                        color='white')

    # Update cumulative values for the next game
    cumulative_ratings += np.array(game_ratings_values)
    cumulative_lower_errors += np.array(lower_bounds)
    cumulative_upper_errors += np.array(upper_bounds)

# Add logos on top of each bar and total values
for i, player in enumerate(sorted_players):
    if player in LOGO_MAPPING:
        logo_path = LOGO_MAPPING[player]
        logo_image = get_logo(logo_path, size=0.6)  # Increased size for better visibility
        if logo_image:
            # Position the logo at the top of the bar
            ab = AnnotationBbox(
                logo_image,
                (i, cumulative_ratings[i]),
                xybox=(0, 35),  # Reduced offset above the bar
                xycoords='data',
                boxcoords="offset points",
                frameon=False
            )
            ax.add_artist(ab)


# Add a horizontal line at y=0
ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.3)  # Increased linewidth

# Customize the grid
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
ax.set_axisbelow(True)  # Put grid behind bars

# Set title and labels with enhanced typography
# ax.set_title('Cumulative Ratings', fontsize=22, fontweight='bold', pad=15)  # Increased size, reduced padding
# ax.set_xlabel('Model', fontsize=16, labelpad=15)
ax.set_ylabel('Rating', fontsize=18, fontweight='bold', labelpad=10)  # Increased size, reduced padding

# Format x-axis labels
plt.xticks(index, [p.replace('-', '\n') for p in sorted_players], rotation=0, ha='center', fontsize=14, fontweight='bold')  # Increased size and made bold

# Format y-axis with fewer ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))  # Reduced number of ticks
plt.yticks(fontsize=14, fontweight='bold')  # Increased size and made bold

# Increase the y-axis limit by 2% (reduced from 3%)
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.02)

# Add a subtle box around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#DDDDDD')
    spine.set_linewidth(0.5)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Reduced from 0.2

# Save the figure with high resolution (PDF only)
plt.savefig('paper/figures/model_performance_comparison.pdf', dpi=300, bbox_inches='tight')

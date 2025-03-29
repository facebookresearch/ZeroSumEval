import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from get_stats import get_all_stats
from matplotlib.offsetbox import AnnotationBbox
from utils import ROOT_DIR, CUSTOM_COLORS, LOGO_MAPPING, get_logo

# Set the style to a more modern look
plt.style.use('seaborn-v0_8-whitegrid')

# Font settings for a more professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'font.weight': 'bold'
})


def plot_combined_performance(games_data, output_dir):
    """Create a combined figure with all performance metrics."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Plot each performance metric in a subplot
    plot_chess_performance(games_data['chess'], axes[0, 0])
    plot_gandalf_performance(games_data['gandalf'], axes[0, 1])
    plot_mathquiz_performance(games_data['mathquiz'], axes[1, 0])
    plot_poker_performance(games_data['poker'], axes[1, 1])

    # Reduce space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outcomes_combined.pdf'), bbox_inches='tight')
    plt.close()

def plot_chess_performance(chess_data, ax):
    """Plot chess-specific performance metrics."""
    models = list(chess_data.keys())
    
    # Extract data for plotting
    wins_by_max_attempts = [chess_data[model]['wins_by_max_attempts'] for model in models]
    wins_by_checkmate = [chess_data[model].get('wins_by_checkmate', 0) for model in models]
    
    # Normalize the data
    total_games_per_model = [wins_by_max_attempts[i] + chess_data[model]['loss_by_max_attempts'] 
                            for i, model in enumerate(models)]
    
    # Calculate percentages for sorting
    sorting_data = []
    for i, model in enumerate(models):
        if total_games_per_model[i] > 0:
            checkmate_pct = wins_by_checkmate[i] / total_games_per_model[i] * 100
            max_attempts_pct = wins_by_max_attempts[i] / total_games_per_model[i] * 100
        else:
            checkmate_pct = 0
            max_attempts_pct = 0
        # Create tuple for sorting: (model, checkmate_pct, max_attempts_pct)
        sorting_data.append((model, checkmate_pct, max_attempts_pct))
    
    # Sort models first by checkmate percentage, then by max attempts percentage
    sorted_models = [x[0] for x in sorted(sorting_data, key=lambda x: (x[1], x[2]), reverse=True)]
    
    # Reorder data based on sorted models
    wins_by_max_attempts = [chess_data[model]['wins_by_max_attempts'] for model in sorted_models]
    wins_by_checkmate = [chess_data[model].get('wins_by_checkmate', 0) for model in sorted_models]
    total_games_per_model = [wins_by_max_attempts[i] + chess_data[model]['loss_by_max_attempts'] 
                            for i, model in enumerate(sorted_models)]
    
    normalized_wins_by_max = [wins_by_max_attempts[i] / total_games_per_model[i] * 100 if total_games_per_model[i] > 0 else 0 
                             for i in range(len(sorted_models))]
    normalized_wins_by_checkmate = [wins_by_checkmate[i] / total_games_per_model[i] * 100 if total_games_per_model[i] > 0 else 0 
                                   for i in range(len(sorted_models))]
    
    # Create the bar plot
    ax.clear()
    ax.set_facecolor('white')
    
    # Plot normalized wins
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, normalized_wins_by_max, width, label='Wins by Max Attempts (%)', color=CUSTOM_COLORS[0])
    ax.bar(x + width/2, normalized_wins_by_checkmate, width, label='Wins by Checkmate (%)', color=CUSTOM_COLORS[1])
    
    ax.set_xlabel('Models', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Win Rate (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title('(A) Chess: Win Type Rates', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.98))
    
    # Set y-axis limit to 120% maximum
    ax.set_ylim(0, 130)
    
    # Add logos on top of each bar with more space
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(normalized_wins_by_max[i], normalized_wins_by_checkmate[i])), 
                    xybox=(0, 25),  # Increased vertical offset
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)

def plot_gandalf_performance(gandalf_data, ax):
    """Plot Gandalf-specific performance metrics."""
    models = list(gandalf_data.keys())
    
    # Sort models by infiltrator wins
    infiltrator_win_counts = []
    for model in models:
        stats = gandalf_data[model]
        infiltrator_wins = stats['infiltrator_wins']
        infiltrator_win_counts.append((model, infiltrator_wins))
    
    sorted_models = [x[0] for x in sorted(infiltrator_win_counts, key=lambda x: x[1], reverse=True)]
    
    # Extract data for plotting
    sentinel_wins = [gandalf_data[model]['sentinel_wins'] for model in sorted_models]
    infiltrator_wins = [gandalf_data[model]['infiltrator_wins'] for model in sorted_models]
    
    # Normalize the data
    total_games_per_model = [sentinel_wins[i] + infiltrator_wins[i] for i in range(len(sorted_models))]
    normalized_sentinel_wins = [sentinel_wins[i] / total_games_per_model[i] * 100 if total_games_per_model[i] > 0 else 0 
                               for i in range(len(sorted_models))]
    normalized_infiltrator_wins = [infiltrator_wins[i] / total_games_per_model[i] * 100 if total_games_per_model[i] > 0 else 0 
                                  for i in range(len(sorted_models))]
    
    # Create the plot
    ax.clear()
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, normalized_sentinel_wins, width, label='Sentinel Wins (%)', color=CUSTOM_COLORS[2])
    ax.bar(x + width/2, normalized_infiltrator_wins, width, label='Infiltrator Wins (%)', color=CUSTOM_COLORS[3])
    
    ax.set_xlabel('Models', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Win Rate (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title('(B) Gandalf: Win Type Rates', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Set y-axis limit to 130% maximum
    ax.set_ylim(0, 140)
    
    # Add logos on top of each bar with more space
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(normalized_sentinel_wins[i], normalized_infiltrator_wins[i])), 
                    xybox=(0, 25),  # Increased vertical offset
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)

def plot_mathquiz_performance(mathquiz_data, ax):
    """Plot MathQuiz-specific performance metrics."""
    models = list(mathquiz_data.keys())
    
    # Extract data for plotting
    correct_answers = [mathquiz_data[model]['wins_by_student_correct_answer'] for model in models]
    verification_failed = [mathquiz_data[model]['wins_by_verification_failed'] for model in models]
    incorrect_answers = [mathquiz_data[model]['wins_by_student_incorrect_answer'] for model in models]
    
    # Normalize the data
    total_outcomes_per_model = [correct_answers[i] + verification_failed[i] + incorrect_answers[i] 
                               for i in range(len(models))]
    
    # Calculate verification failed percentages for sorting
    verification_failed_percentages = []
    for i, model in enumerate(models):
        if total_outcomes_per_model[i] > 0:
            verification_pct = verification_failed[i] / total_outcomes_per_model[i] * 100
        else:
            verification_pct = 0
        verification_failed_percentages.append((model, verification_pct))
    
    # Sort models by verification failed percentage (increasing order)
    sorted_models = [x[0] for x in sorted(verification_failed_percentages, key=lambda x: x[1])]
    
    # Reorder data based on sorted models
    correct_answers = [mathquiz_data[model]['wins_by_student_correct_answer'] for model in sorted_models]
    verification_failed = [mathquiz_data[model]['wins_by_verification_failed'] for model in sorted_models]
    incorrect_answers = [mathquiz_data[model]['wins_by_student_incorrect_answer'] for model in sorted_models]
    
    total_outcomes_per_model = [correct_answers[i] + verification_failed[i] + incorrect_answers[i] 
                               for i in range(len(sorted_models))]
    
    normalized_correct = [correct_answers[i] / total_outcomes_per_model[i] * 100 if total_outcomes_per_model[i] > 0 else 0 
                         for i in range(len(sorted_models))]
    normalized_verification = [verification_failed[i] / total_outcomes_per_model[i] * 100 if total_outcomes_per_model[i] > 0 else 0 
                              for i in range(len(sorted_models))]
    normalized_incorrect = [incorrect_answers[i] / total_outcomes_per_model[i] * 100 if total_outcomes_per_model[i] > 0 else 0 
                           for i in range(len(sorted_models))]
    
    # Create the plot
    ax.clear()
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.25
    
    ax.bar(x - width, normalized_correct, width, label='Student Correct (%)', color=CUSTOM_COLORS[4])
    ax.bar(x, normalized_verification, width, label='Verification Failed (%)', color=CUSTOM_COLORS[5])
    ax.bar(x + width, normalized_incorrect, width, label='Student Incorrect (%)', color=CUSTOM_COLORS[6])
    
    ax.set_xlabel('Models', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Rate (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title('(C) MathQuiz: Outcome Type Rates', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.98))
    
    # Set y-axis limit to 120% maximum
    ax.set_ylim(0, 130)
    
    # Add logos on top of each bar with more space
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(normalized_correct[i], normalized_verification[i], normalized_incorrect[i])), 
                    xybox=(0, 25),  # Increased vertical offset
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)

def plot_poker_performance(poker_data, ax):
    """Plot Poker-specific performance metrics."""
    models = list(poker_data.keys())
    
    # Calculate average winning chip differences
    avg_chip_diffs = []
    for model in models:
        stats = poker_data[model]
        avg_diff = np.mean(stats['winning_chip_differences']) if stats['winning_chip_differences'] else 0
        avg_chip_diffs.append((model, avg_diff))
    
    sorted_models = [x[0] for x in sorted(avg_chip_diffs, key=lambda x: max(poker_data[x[0]]['winning_chip_differences']) if poker_data[x[0]]['winning_chip_differences'] else 0, reverse=True)]
    
    # Extract data for plotting
    avg_chips = [np.mean(poker_data[model]['winning_chip_differences']) for model in sorted_models]
    max_chips = [max(poker_data[model]['winning_chip_differences']) if poker_data[model]['winning_chip_differences'] else 0 for model in sorted_models]
    
    # Find the maximum chip value across all models for normalization
    max_chip_value = max([max(max_chips), max(avg_chips)])
    
    # Normalize the data (as percentage of maximum value)
    normalized_avg_chips = [avg / max_chip_value * 100 if max_chip_value > 0 else 0 for avg in avg_chips]
    normalized_max_chips = [max_val / max_chip_value * 100 if max_chip_value > 0 else 0 for max_val in max_chips]
    
    # Create the plot
    ax.clear()
    ax.set_facecolor('white')
    
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax.bar(x - width/2, normalized_avg_chips, width, label='Avg Chips (% of Max)', color=CUSTOM_COLORS[0])
    ax.bar(x + width/2, normalized_max_chips, width, label='Max Chips (% of Max)', color=CUSTOM_COLORS[1])
    
    ax.set_xlabel('Models', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Chip Difference (%)', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title('(D) Poker: Normalized Chip Differences', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.98))
    
    # Set y-axis limit to 120% maximum
    ax.set_ylim(0, 130)
    
    # Add logos on top of each bar with more space
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                ab = AnnotationBbox(
                    logo_image, 
                    (i, max(normalized_avg_chips[i], normalized_max_chips[i])), 
                    xybox=(0, 25),  # Increased vertical offset
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)

def plot_chess_moves_histogram(chess_data, output_dir):
    """Plot violin plots of the number of moves in chess games for each model."""
    models = list(chess_data.keys())
    
    # Prepare data for violin plot
    data = [chess_data[model]['num_moves'] for model in models]
    
    # Calculate range (max - min) for each model to sort by distribution size
    range_values = []
    for i, model in enumerate(models):
        model_data = data[i]
        if model_data and len(model_data) > 1:
            data_range = np.max(model_data) - np.min(model_data)
            range_values.append((model, data_range, model_data))
        else:
            range_values.append((model, 0, model_data))
    
    # Sort models by range (distribution size)
    sorted_data = sorted(range_values, key=lambda x: x[1], reverse=True)
    sorted_models = [item[0] for item in sorted_data]
    data = [item[2] for item in sorted_data]
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create violin plot
    violin_parts = ax.violinplot(data, showmeans=True, showmedians=True)
    
    # Customize violin colors
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(CUSTOM_COLORS[i % len(CUSTOM_COLORS)])
        pc.set_edgecolor('#333333')
        pc.set_alpha(0.7)
    
    # Customize other parts
    violin_parts['cmeans'].set_color('#111111')
    violin_parts['cmedians'].set_color('#333333')
    violin_parts['cbars'].set_color('#333333')
    violin_parts['cmins'].set_color('#333333')
    violin_parts['cmaxes'].set_color('#333333')
    
    # Calculate statistics for each dataset to position logos
    positions = []
    for i, model_data in enumerate(data):
        if model_data:
            # Calculate the position for the logo
            max_val = np.max(model_data)  # Maximum value
            positions.append((i+1, max_val))  # Violin positions are 1-indexed
        else:
            positions.append((i+1, 0))
    
    # Add logos above each violin
    for i, model in enumerate(sorted_models):
        if model in LOGO_MAPPING:
            logo_path = LOGO_MAPPING[model]
            logo_image = get_logo(logo_path, size=0.3)
            if logo_image:
                violin_pos, max_y = positions[i]
                
                ab = AnnotationBbox(
                    logo_image, 
                    (violin_pos, max_y), 
                    xybox=(0, 25),
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=False
                )
                ax.add_artist(ab)
    
    # Set x-tick labels
    ax.set_xticks(np.arange(1, len(sorted_models) + 1))
    ax.set_xticklabels(sorted_models, fontsize=14, fontweight='bold')
    
    # Set y-axis limit to 60
    ax.set_ylim(0, 60)
    
    ax.set_title('Valid Chess Move Distributions', fontsize=22, fontweight='bold', pad=15)
    ax.set_xlabel('Models', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Moves', fontsize=18, fontweight='bold', labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Customize the grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#333333')
    ax.set_axisbelow(True)  # Put grid behind violins
    
    # Add a subtle box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDDDDD')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chess_moves_violin.pdf'), bbox_inches='tight')
    plt.close()

def main():
    output_dir = "paper/figures/"
    os.makedirs(output_dir, exist_ok=True)
    games_data = get_all_stats(ROOT_DIR)
    
    # Generate visualizations as PDFs
    plot_combined_performance(games_data, output_dir)
    plot_chess_moves_histogram(games_data['chess'], output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()

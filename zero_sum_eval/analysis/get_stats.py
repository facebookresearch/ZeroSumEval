# chess - wins because max attempts, wins because checkmate, per model histogram of number of moves
# debate - win because max attempts, regular win
# gandalf - wins because max attempts, infiltrator win vs sentinel win
# liars dice - win because max attempts, regular wins
# mathquiz - student win because answer correct, student win because verification failed, teacher win because student wrong
# poker - wins due to max attempts, chip difference across wins
# pyjail - attacker win because attacker win because verification failed
from collections import defaultdict, namedtuple
import json
import os
import jsonlines
import glob

import yaml

Match = namedtuple("Match", ["models", "turns", "scores"])

def get_matches(path):
    """
    returns a generator of Match objects
    """
    paths = glob.glob(f"{path}/matches/*")
    for p in paths:
        if not os.path.exists(os.path.join(p, "turns.jsonl")):
            continue
        with jsonlines.open(os.path.join(p, "turns.jsonl")) as f:
            turns = [turn for turn in f]
        
        with open(os.path.join(p, "scores.json")) as f:
            scores = json.load(f)
        
        models = p.split("/")[-1].split("_vs_")
        models[1] = models[1].split("_")[0]
        yield Match(models, turns, scores)

def get_max_attempts_wl(path):
    """
    returns {
     "llama": {
        "wins_by_max_attempts": <int>,
        "loss_by_max_attempts": <int>,
     },
     ...
    """
    matches = get_matches(path)
    with open(os.path.join(path, "pool_config.yaml"), "r") as f:
        pool_config = yaml.safe_load(f)
    max_player_attempts = pool_config["manager"]["max_player_attempts"]

    stats = {}

    for match in matches:
        for model, scores in match.scores.items():
            if scores["attempts"] >= max_player_attempts:
                # get other model
                stats[model] = stats.get(model, {"wins_by_max_attempts": 0, "loss_by_max_attempts": 0})
                stats[model]["loss_by_max_attempts"] += 1
                winner_model = [m for m in match.models if m != model][0]
                stats[winner_model] = stats.get(winner_model, {"wins_by_max_attempts": 0, "loss_by_max_attempts": 0})
                stats[winner_model]["wins_by_max_attempts"] += 1

    return stats

def get_chess_stats(path):
    """
    returns {
     "llama": {
        wins_by_max_attempts: <int>,
        wins_by_checkmate: <int>,
        num_moves: [...]
     },
     ...
    """
    matches = get_matches(path)
    stats = defaultdict(lambda: {"wins_by_checkmate": 0, "num_moves": []})

    for match in matches:
        stats[match.models[0]]["num_moves"].append(len(match.turns) // 2)
        if match.turns[-1]["message"] == "Checkmate":
            stats[match.models[0]]["wins_by_checkmate"] += 1

    return stats

def get_role_wins_stats(path):
    """
    for example, in gandalf:
    {
     "llama": {
        "infiltrator_wins": <int>,
        "sentinel_wins": <int>,
     },
     ...
    """
    matches = get_matches(path)
    stats = {}
    with open(os.path.join(path, "pool_config.yaml"), "r") as f:
        pool_config = yaml.safe_load(f)

    roles = pool_config["game"]["args"]["players"].keys()
    stats = defaultdict(lambda: {f"{role}_wins": 0 for role in roles})


    for match in matches:
        for model, scores in match.scores.items():
            if scores["score"] == 1:
                stats[model][f"{scores['role']}_wins"] += 1

    return stats
        

def get_mathquiz_stats(path):
    """
    returns {
     "llama": {
        wins_by_student_correct_answer: <int>,
        wins_by_verification_failed: <int>,
        wins_by_student_incorrect_answer: <int>,
     },
     ...
    """
    matches = get_matches(path)
    stats = defaultdict(lambda: {"wins_by_student_correct_answer": 0, "wins_by_verification_failed": 0, "wins_by_student_incorrect_answer": 0})
    for match in matches:
        for model, scores in match.scores.items():
            
            if scores["score"] == 1:
                if scores["role"] == "student":
                    if match.turns[-1]["student_answer"] is not None:
                        stats[model]["wins_by_student_correct_answer"] += 1
                    else:
                        stats[model]["wins_by_verification_failed"] += 1
                elif scores["role"] == "teacher":
                    if match.turns[-1]["teacher_answer"] is not None:
                        stats[model]["wins_by_student_incorrect_answer"] += 1
    return stats

def get_poker_stats(path):
    """
    returns {
     "llama": {
        winning_chip_differences: [...]
     },
     ...
    """
    matches = get_matches(path)
    stats = defaultdict(lambda: {"winning_chip_differences": []})
    for match in matches:
        score_1 = match.scores[match.models[0]]["score"]
        score_2 = match.scores[match.models[1]]["score"] 
        if score_1 > score_2:
            stats[match.models[0]]["winning_chip_differences"].append(score_1 - score_2)
        else:
            stats[match.models[1]]["winning_chip_differences"].append(score_2 - score_1)

    return stats


filenames = {
    "chess": "rankings-3-9-25_chess",
    "debate": "rankings-3-9-25_debate",
    "gandalf": "rankings-3-9-25_gandalf_final_500",
    "liars_dice": "rankings-3-9-25_liars_dice",
    "mathquiz": "rankings-3-9-25_mathquiz_final_500",
    "poker": "rankings-3-9-25_poker_final_500",
}

if __name__ == "__main__":
    results_path = "results"
    for game, filename in filenames.items():
        max_attempts_wl = get_max_attempts_wl(os.path.join(results_path, filename))
        stats = {model: {**max_attempts_wl[model], **stats[model]} for model in stats}
        if game == "chess":
            stats = get_chess_stats(os.path.join(results_path, filename))
            # add max attempts wins and losses
            stats = {model: {**max_attempts_wl[model], **stats[model]} for model in stats}
            # TODO: visualization
        elif game == "mathquiz":
            stats = get_mathquiz_stats(os.path.join(results_path, filename))
            # add max attempts wins and losses
            stats = {model: {**max_attempts_wl[model], **stats[model]} for model in stats}
            # TODO: visualization
        elif game == "poker":
            stats = get_poker_stats(os.path.join(results_path, filename))
            # add max attempts wins and losses
            stats = {model: {**max_attempts_wl[model], **stats[model]} for model in stats}
            # TODO: visualization
        elif game == "pyjail":
            # TODO: get pyjail stats
            pass
        else:
            stats = get_role_wins_stats(os.path.join(results_path, filename))
            # add max attempts wins and losses
            stats = {model: {**max_attempts_wl[model], **stats[model]} for model in stats}
            # TODO: visualization
        print(stats)

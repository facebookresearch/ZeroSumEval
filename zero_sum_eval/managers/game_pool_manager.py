# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import json
from logging import getLogger
from math import inf
import os
import time
from typing import Dict, List, Optional

from zero_sum_eval.managers.game_manager import GameManager
from zero_sum_eval.managers.matchers import RoundRobin
from zero_sum_eval.core.registry import GAME_REGISTRY

WDL_FILE = "wdl.json"
SCORES_FILE = "scores.json"
MATCH_DIR = "matches/"

class GamePoolManager:

    def __init__(
        self,
        max_matches: int = 10,
        max_concurrent_matches: int = 1,
        max_rounds_per_match: int = 100,
        max_player_attempts: int = 5,
        max_time_per_player: Optional[float] = None,
        output_dir: str = "pool_output",
        game: str = "chess",
        game_args: Dict = {},
        llm_configs: List[Dict] = [
            {
                "name": "gpt-4o",
                "model": "openrouter/openai/chatgpt-4o-latest",
            },
            {
                "name": "claude-3-5-sonnet",
                "model": "openrouter/claude-3-5-sonnet-20240620",
            },
            {
                "name": "mistral-large",
                "model": "openrouter/mistralai/mistral-large-2411",
            }
        ],
    ):
        self.logger = getLogger()
        self.max_matches = max_matches
        self.max_concurrent_matches = max_concurrent_matches
        self.max_rounds_per_match = max_rounds_per_match
        self.max_player_attempts = max_player_attempts
        self.max_time_per_player = max_time_per_player
        self.output_dir = output_dir
        self.prev_matches_played = 0

        # Ensure unique names for LMs
        seen = defaultdict(int)
        for llm in llm_configs:
            if llm["name"] in seen:
                seen[llm["name"]] += 1
                llm["name"] += f"_{seen[llm['name']]}"
            else:
                seen[llm["name"]] = 1
        self.llm_configs = {llm["name"]: {k: v for k, v in llm.items() if k != "action_args"} for llm in llm_configs}
        self.action_args = {llm["name"]: llm.get("action_args", {}) for llm in llm_configs}
        self.llm_wdl = {llm_name: {"wins": 0, "draws": 0, "losses": 0} for llm_name in self.llm_configs}

        self.game = game
        self.game_args = game_args

        self.matcher = RoundRobin(
            # RoundRobin does not use ELO ratings for matching, so all ELOs are set arbitrarily to -1
            llm_elos={llm["name"]: -1 for llm in llm_configs},
            players_per_match=len(self._get_player_configs()),
        )
        self.logger.info(f"Scheduled matches: {self.matcher.matches}")

        self.match_freq = dict()
        for matchup in self.matcher.matches:
            self.match_freq[matchup] = 0
        for match in glob(f"{self.output_dir}/{MATCH_DIR}/*"):
            if os.path.exists(os.path.join(match, SCORES_FILE)):
                with open(os.path.join(match, SCORES_FILE), mode='r', newline='') as f:
                    scores = json.load(f)
                self.calculate_wdl(scores)
                model_names = match.split("/")[-1].split("_vs_")
                model_names[-1] = model_names[-1].split("_")[0]
                model_names = tuple(model_names)
                if model_names in self.match_freq:
                    self.match_freq[model_names] += 1
                    self.prev_matches_played += 1
        self.logger.info(f"Match frequency:")
        for matchup, freq in self.match_freq.items():
            self.logger.info(f"{matchup[0]} vs {matchup[1]}: {freq}")

    def _get_player_configs(self):
        player_definitions = GAME_REGISTRY[self.game.lower()].player_definitions()
        player_configs = self.game_args.get("players", {}).copy()
        for player_def in player_definitions:
            if player_def.player_key not in self.game_args["players"] and not player_def.optional:
                player_configs[player_def.player_key] = {}
        return player_configs

    def _build_game(self, lms: List[str]):
        player_configs = self._get_player_configs()
        if len(player_configs) != len(lms):
            raise ValueError(f"Number of players {len(player_configs)} must match the number of LMs {len(lms)}")

        for (role, player_config), lm in zip(player_configs.items(), lms):
            if "args" not in player_config:
                player_config["args"] = {}
            player_config["args"]["lm"] = self.llm_configs[lm]
            player_config["args"]["id"] = f"{lm}||{role}"
            player_config["args"]["action_args"] = self.action_args[lm]
        config = {
            **self.game_args,
            "players": player_configs,
        }
        return GAME_REGISTRY.build(self.game, **config)


    def get_next_min_match(self):
        """Gets the match with the least number of games played"""
        return min(self.match_freq.keys(), key=lambda k: self.match_freq[k])

    def run_match(self, lms):
        turn_dir = os.path.join(self.output_dir, MATCH_DIR)
        for lm in lms:
            turn_dir += f"{lm}_vs_"

        turn_dir = turn_dir[:-4]
        turn_dir += f"_{int(time.time())}"
        os.makedirs(turn_dir, exist_ok=True)
        game_state = self._build_game(lms=lms)
        game_manager = GameManager(max_rounds=self.max_rounds_per_match, max_player_attempts=self.max_player_attempts, output_dir=turn_dir, max_time_per_player=self.max_time_per_player)

        matchup_string = ""
        for player in game_state.players.values():
            lm, role = player.id.split("||")
            matchup_string += f"{lm} as {role} VS "
        matchup_string = matchup_string[:-4]
        self.logger.info(f"Matchup: {matchup_string}")

        results = game_manager.start(game_state)
        final_game_state = results["game_state"]
        player_attempts = results["player_attempts"]
        player_time_used = results["player_time_used"]

        winner_role = max(final_game_state.get_scores().items(), key=lambda x: x[1])[0]

        # Get the winner's name without the role (ids look like "{lm}||{role}", for example "gpt-4o||white")
        winner = final_game_state.players[winner_role].id.split("||")[0]

        self.logger.info(f"{winner} won! Final scores: {final_game_state.get_scores()}")

        scores = {}
        for role, player in final_game_state.players.items():
            lm, role = player.id.split("||")
            scores[lm] = {
                "score": final_game_state.get_scores()[role],
                "role": role,
                "attempts": player_attempts[player],
                "total_time": player_time_used[player],
            }

        with open(os.path.join(turn_dir, SCORES_FILE), mode='w', newline='') as f:
            json.dump(scores, f)

        return scores

    def calculate_wdl(self, scores):
        scores = {lm: result["score"] if result["attempts"] < self.max_player_attempts else -inf for lm, result in scores.items()}
        best_score = max(scores.items(), key=lambda x: x[1])[1]
        best_lms = [lm for lm, score in scores.items() if score == best_score]

        for lm, score in scores.items():
            if lm in self.llm_wdl:
                if score == best_score:
                    if len(best_lms) > 1:
                        self.llm_wdl[lm]["draws"] += 1
                    else:
                        self.llm_wdl[lm]["wins"] += 1
                else:
                    self.llm_wdl[lm]["losses"] += 1

    def start(self):
        self.logger.info("Let the games begin!")
        if self.prev_matches_played >= self.max_matches:
            self.logger.info(f"No more matches to play, to continue this pool, set max_matches > {self.prev_matches_played=}")
            return self.llm_wdl

        with ThreadPoolExecutor(max_workers=self.max_concurrent_matches) as executor:
            future_to_match = {}
            for _ in range(self.max_matches - self.prev_matches_played):
                lms = self.get_next_min_match()
                future = executor.submit(self.run_match, lms)
                self.match_freq[tuple(lms)] += 1
                future_to_match[future] = lms

            for future in as_completed(future_to_match):
                try:
                    scores = future.result()
                except Exception as e:
                    for f in future_to_match:
                        f.cancel()
                    raise e

                self.calculate_wdl(scores)

        with open(os.path.join(self.output_dir, WDL_FILE), mode='w', newline='') as f:
            json.dump(self.llm_wdl, f)

        return self.llm_wdl

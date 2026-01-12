
# Copyright (c) Meta Platforms, Inc. and affiliates.

from zero_sum_eval.core.player import Player

WHITE_KEY = "White"
BLACK_KEY = "Black"

class CheckersPlayer(Player):
    """
    A simple player wrapper for Checkers.
    Most logic is handled by the GameState or LLM, this class primarily
    serves as the agent interface.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

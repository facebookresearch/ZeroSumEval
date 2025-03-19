# I took inspiration from https://github.com/carlini/chess-llm and https://github.com/mlabonne/chessllm
# Shout out to the maintainers and authors of these repositories!

from zero_sum_eval.core.player import Player
import dspy
import chess
from chess import IllegalMoveError, InvalidMoveError, AmbiguousMoveError
from zero_sum_eval.core.registry import PLAYER_REGISTRY, METRIC_REGISTRY
from stockfish import Stockfish

# TODO: add support for resigning

# Player keys
WHITE_KEY = "white"
BLACK_KEY = "black"


@METRIC_REGISTRY.register("chess_move_validation_metric")
def validate_move(example, prediction, trace=None):
    pred_move = prediction.move
    board_state = example.board_state
    board = chess.Board(board_state)
    try:
        if board.is_legal(board.parse_san(pred_move)):
            return 1
        return 0
    except (IllegalMoveError, InvalidMoveError, AmbiguousMoveError):
        return 0
    
@METRIC_REGISTRY.register("chess_stockfish_metric")
def stockfish_metric(example: dspy.Example, prediction: dspy.Example, trace=None, margin=5):
    board_state = example.board_state
    pred_move = prediction.move
    board = chess.Board(board_state)
    try:
        if not board.is_legal(board.parse_san(pred_move)):
            return 0
    except (IllegalMoveError, InvalidMoveError, AmbiguousMoveError):
        return 0
    
    stockfish = Stockfish("/usr/games/stockfish", parameters={"Threads": 1, "Minimum Thinking Time": 1000})
    is_white = board.turn
    stockfish.set_fen_position(board.fen())
    eval_prev = stockfish.get_evaluation()
    board.push_san(pred_move)
    stockfish.set_fen_position(board.fen())
    eval_after = stockfish.get_evaluation()
    if is_white:
        return eval_after["value"] > eval_prev["value"] - margin
    else:
        return eval_prev["value"] > eval_after["value"] - margin  

class NextMove(dspy.Signature):
    """Given a board state, role, and move history, produce the next best valid move"""
    board_state = dspy.InputField(desc="FEN formatted current board state")
    role = dspy.InputField(desc="role of the player making the next move")
    history = dspy.InputField(desc="move history")
    move = dspy.OutputField(desc="a valid SAN move without move number, elipses, or any extra formatting.")

class ChessModule(dspy.Module):
    def __init__(self, module):
        self.move = module(NextMove)

    def forward(self, board_state, role, history):
        out = self.move(
            board_state=board_state,
            role=role,
            history=history
        )
        out.move = out.move.replace(".", "")
        try:
            board = chess.Board(board_state)
            board.parse_san(out.move)
        except (IllegalMoveError, InvalidMoveError, AmbiguousMoveError) as e:
            error_messages = {
                IllegalMoveError: "illegal",
                InvalidMoveError: "invalid",
                AmbiguousMoveError: "ambiguous"
            }
            error_type = type(e)
            dspy.Suggest(
                False,
                f"{out.move} is an {error_messages[error_type]} move, choose a different move."
            )
        return out


@PLAYER_REGISTRY.register("chess", "chess_player")
class ChessPlayer(Player):
    def init_actions(self, module: str = "ChainOfThought"):
        supported_modules = {
            "ChainOfThought": dspy.ChainOfThought,
            "Predict": dspy.Predict,            
        }
        if module not in supported_modules:
            raise ValueError(f"Module {module} not supported, supported modules are: {supported_modules.keys()}")

        return {
            "MakeMove": ChessModule(module=supported_modules[module])
        }
    
@PLAYER_REGISTRY.register("chess", "human_player")
class HumanPlayer(Player):
    def init_actions(self):
        return {
            "MakeMove": lambda **args: input("Enter a move: ")
        }

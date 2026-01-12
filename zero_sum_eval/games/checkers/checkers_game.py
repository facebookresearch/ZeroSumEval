
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
from typing import List, Tuple, Dict, Optional

from zero_sum_eval.core.game_state import GameState, InvalidMoveError, PlayerDefinition
from zero_sum_eval.core.player import Move
from zero_sum_eval.utils.types import Action
from zero_sum_eval.core.registry import GAME_REGISTRY
from zero_sum_eval.games.checkers.checkers_player import CheckersPlayer, WHITE_KEY, BLACK_KEY

# Constants for board representation
EMPTY = 0
WHITE = 1
BLACK = 2
WHITE_KING = 3
BLACK_KING = 4

@GAME_REGISTRY.register("checkers")
class CheckersGame(GameState):
    """
    Standard US Checkers (English Draughts) implementation.
    Board is 8x8.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.board = self._init_board()
        self.turn = WHITE_KEY 
        self.history = []
        self.scores = {WHITE_KEY: 0.0, BLACK_KEY: 0.0}
        self.message = f"{WHITE_KEY} to move."
        self.consecutive_non_capture_moves = 0 # For draw rules (simplified)

    def _init_board(self) -> List[List[int]]:
        """
        Initialize standard 8x8 checkers board.
        White at bottom (rows 5,6,7), Black at top (rows 0,1,2).
        Dark squares only.
        """
        board = [[EMPTY] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    if r < 3:
                        board[r][c] = BLACK
                    elif r > 4:
                        board[r][c] = WHITE
        return board

    def get_scores(self) -> Dict[str, float]:
        return self.scores

    def is_over(self) -> bool:
        # Game over if current player has no moves or no pieces
        moves = self._get_valid_moves(self.turn)
        if not moves:
            return True
        # Also could check for draws, but adhering to basic rules for now
        return False

    def update_game(self, move: Move) -> 'CheckersGame':
        # Move string format: "r1,c1 -> r2,c2" or "r1,c1 -> r2,c2 -> r3,c3" (multijump)
        # Coordinate system: 0,0 is top-left.
        try:
            target_move = None
            valid_moves = self._get_valid_moves(self.turn)
            
            # Simple parsing/matching logic could be improved 
            # We assume the move string matches exactly one of our generated move strings
            for vm in valid_moves:
                if vm["move_str"] == move.value:
                    target_move = vm
                    break
            
            if not target_move:
                # Try to fuzzy match or re-parse if needed, but for now strict
                raise InvalidMoveError(f"Move {move.value} is not valid for {self.turn}")

            # Execute move
            self._apply_move(target_move)
            
            # Record history
            self.history.append(move.value)
            
            # Check game end immediately after move
            next_turn = BLACK_KEY if self.turn == WHITE_KEY else WHITE_KEY
            next_valid_moves = self._get_valid_moves(next_turn)
            
            if not next_valid_moves:
                # Current player just won
                self.message = f"Winner: {self.turn}"
                self.scores[self.turn] = 1.0
                self.scores[next_turn] = 0.0
                # Do NOT switch turn so is_over returns true correctly (or handle in is_over)
                # Actually is_over checks for current turn moves. So we need to switch turn to the loser who has no moves-
                self.turn = next_turn 
            else:
                self.turn = next_turn
                self.message = f"{self.turn} to move."

        except Exception as e:
            if isinstance(e, InvalidMoveError):
                raise e
            raise InvalidMoveError(f"Error executing move {move.value}: {str(e)}")
        
        return self

    def get_next_action(self) -> Action:
        inputs = {
            'board_state': self.export_board(),
            'role': self.message,
            'history': self.formatted_move_history(),
            'valid_moves': [m['move_str'] for m in self._get_valid_moves(self.turn)]
        }
        # In this framework, we assume standard alternating turns
        return Action(name="MakeMove", player_key=self.turn, inputs=inputs)

    def _get_valid_moves(self, player) -> List[Dict]:
        """
        Returns list of dicts: {'move_str': str, 'steps': [...], 'captures': [...]}
        steps is list of (r,c) tuples.
        """
        moves = []
        piece_color = WHITE if player == WHITE_KEY else BLACK
        king_color = WHITE_KING if player == WHITE_KEY else BLACK_KING
        opponent_color = BLACK if player == WHITE_KEY else WHITE
        opponent_king = BLACK_KING if player == WHITE_KEY else WHITE_KING
        
        pieces = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p == piece_color or p == king_color:
                    pieces.append((r,c))
        
        # 1. Check for jumps (captures) - Mandatory in many rules, let's enforce mandatory jumps for simplicity and standard tournament rules
        jumps = []
        for r, c in pieces:
            jumps.extend(self._get_jumps(r, c, self.board[r][c]))
            
        if jumps:
             # If jumps exist, you MUST jump. (Standard rules)
             # Filter primarily for longest jumps? Or just any capture? 
             # US rules: capturing is mandatory, but choosing *which* capture sequence is player's choice. 
             # We will return all valid capture chains.
             moves = jumps
        else:
            # 2. Simple moves
            for r, c in pieces:
                moves.extend(self._get_simple_moves(r, c, self.board[r][c]))
                
        # Format moves
        formatted = []
        for m in moves:
             # m is sequence of coords [(start_r, start_c), (next_r, next_c), ...]
             move_str = " -> ".join([f"({r},{c})" for r,c in m['path']])
             m['move_str'] = move_str
             formatted.append(m)
             
        return formatted

    def _get_simple_moves(self, r, c, piece) -> List[Dict]:
        moves = []
        directions = []
        if piece == WHITE: 
            directions = [(-1, -1), (-1, 1)] # Up
        elif piece == BLACK: 
            directions = [(1, -1), (1, 1)] # Down
        else: # King
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.board[nr][nc] == EMPTY:
                    moves.append({
                        'path': [(r,c), (nr,nc)],
                        'captures': []
                    })
        return moves

    def _get_jumps(self, r, c, piece) -> List[Dict]:
        # DFS for multi-jumps
        jumps = []
        stack = [(r, c, piece, [(r, c)], [])] # current_r, current_c, current_piece_state, path, captured_coords
        
        # This is a simplification. We need to handle the board state copy for multi-jumps or marking captured pieces as 'pending removal'
        # Actually easier: recursing with board copies or just calculating geometric possibility.
        # Since board is small, passing copies is fine.
        
        return self._recurse_jumps(r, c, piece, [(r,c)], [], [row[:] for row in self.board])

    def _recurse_jumps(self, r, c, piece, path, captures, board_state) -> List[Dict]:
        directions = []
        is_king = (piece == WHITE_KING or piece == BLACK_KING)
        
        # Determine allowed directions
        if piece == WHITE and not is_king: directions = [(-1, -1), (-1, 1)]
        elif piece == BLACK and not is_king: directions = [(1, -1), (1, 1)]
        else: directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        valid_continuations = []
        
        for dr, dc in directions:
            nr, nc = r + dr * 2, c + dc * 2
            mid_r, mid_c = r + dr, c + dc
            
            if 0 <= nr < 8 and 0 <= nc < 8:
                mid_p = board_state[mid_r][mid_c]
                dest_p = board_state[nr][nc]
                
                # Check for enemy
                is_enemy = False
                if piece in [WHITE, WHITE_KING]:
                    if mid_p in [BLACK, BLACK_KING]: is_enemy = True
                else: # Black
                    if mid_p in [WHITE, WHITE_KING]: is_enemy = True
                
                if is_enemy and dest_p == EMPTY:
                    # Found a jump
                    new_captures = captures + [(mid_r, mid_c)]
                    new_path = path + [(nr, nc)]
                    
                    # Simulate jump on a temp board
                    new_board = [row[:] for row in board_state]
                    new_board[r][c] = EMPTY
                    new_board[mid_r][mid_c] = EMPTY # Temporarily remove captured
                    
                    # Handle promotion during jump?
                    # US Rules: If pawn reaches King row, turn ENDS. No multi-jumps out of promotion.
                    promoted = False
                    new_piece = piece
                    if piece == WHITE and nr == 0:
                        new_piece = WHITE_KING
                        promoted = True
                    elif piece == BLACK and nr == 7:
                        new_piece = BLACK_KING
                        promoted = True
                    
                    new_board[nr][nc] = new_piece
                    
                    if promoted:
                        # Must stop
                        valid_continuations.append({
                            'path': new_path,
                            'captures': new_captures
                        })
                    else:
                        # Continue jumping
                        continuations = self._recurse_jumps(nr, nc, new_piece, new_path, new_captures, new_board)
                        if not continuations:
                            # No more jumps, this is a valid terminal jump
                            valid_continuations.append({
                                'path': new_path,
                                'captures': new_captures
                            })
                        else:
                            valid_continuations.extend(continuations)
        
        return valid_continuations

    def _apply_move(self, move_dict):
        path = move_dict['path']
        start_r, start_c = path[0]
        end_r, end_c = path[-1]
        captures = move_dict['captures']
        
        piece = self.board[start_r][start_c]
        
        # Remove piece from start
        self.board[start_r][start_c] = EMPTY
        
        # Remove captures
        for cr, cc in captures:
            self.board[cr][cc] = EMPTY
            
        # Place at end with promotion check
        if piece == WHITE and end_r == 0:
            piece = WHITE_KING
        elif piece == BLACK and end_r == 7:
            piece = BLACK_KING
            
        self.board[end_r][end_c] = piece

    def formatted_move_history(self) -> str:
        return " ".join(self.history)
        
    def export_board(self) -> str:
        # Simple ASCII representation or FEN-like
        # Using simple list of lists str for now as it's readable
        out = ""
        for row in self.board:
            out += str(row) + "\n"
        return out

    @classmethod
    def player_definitions(cls) -> List[PlayerDefinition]:
        return [
            PlayerDefinition(player_key=WHITE_KEY, actions=["MakeMove"], default_player_class=CheckersPlayer),
            PlayerDefinition(player_key=BLACK_KEY, actions=["MakeMove"], default_player_class=CheckersPlayer)
        ]

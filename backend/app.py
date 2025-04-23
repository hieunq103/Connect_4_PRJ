from fastapi import FastAPI, HTTPException
import random
import uvicorn
import numpy as np
import math
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4
MAX_TABLE_SIZE = 1000000

# Global dictionary for memoization
transposition_table = {}

# Pydantic models
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

# Evaluation functions
def evaluate_window(window, piece, row_index=None, col_start=None, board=None):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    def is_playable(col_offset):
        if board is None or row_index is None or col_start is None:
            return True
        col = col_start + col_offset
        if row_index == len(board) - 1:
            return True
        return board[row_index + 1][col] != EMPTY

    if window.count(piece) == 4:
        score += 100000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        empty_index = window.index(EMPTY)
        if is_playable(empty_index):
            score += 100
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10

    if window.count(opp_piece) == 4:
        score -= 100000
    elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        empty_index = window.index(EMPTY)
        if is_playable(empty_index):
            score -= 80
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 5

    return score

def score_position(board, piece):
    board_array = np.array(board)
    score = 0
    row_count = len(board)
    column_count = len(board[0])
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    # Score center column
    center_array = [int(i) for i in list(board_array[:, column_count//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score horizontal
    for r in range(row_count):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(column_count - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece, row_index=r, col_start=c, board=board)

    # Score vertical
    for c in range(column_count):
        col_array = [int(i) for i in list(board_array[:,c])]
        for r in range(row_count-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(row_count-3):
        for c in range(column_count-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(row_count-3):
        for c in range(column_count-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Prioritize positions by height
    for c in range(column_count):
        for r in range(row_count):
            if board[r][c] == piece:
                score += (r + 1) * 0.5

    # Evaluate multi-directional threats
    for c in range(column_count):
        r = get_next_open_row(board, c)
        if r == -1:  # Column is full
            continue
            
        # Test placing our piece in this position
        board_copy = [row[:] for row in board]
        board_copy[r][c] = piece
        
        threat_directions = 0
        
        # Check horizontal threat
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(piece) == 3 and row_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Check vertical threat
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(piece) == 3 and col_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Check diagonal down-right threat
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Check diagonal down-left threat
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Score multi-directional threats highly
        if threat_directions > 1:
            score += 100 * threat_directions
        
        # Check opponent threats at this position
        board_copy = [row[:] for row in board]
        board_copy[r][c] = opp_piece
        
        opp_threat_directions = 0
        
        # Check horizontal opponent threat
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(opp_piece) == 3 and row_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Check vertical opponent threat
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(opp_piece) == 3 and col_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Check diagonal down-right opponent threat
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Check diagonal down-left opponent threat
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Prioritize defense if opponent has multiple threats
        if opp_threat_directions > 1:
            score -= 120 * opp_threat_directions

    return score

# Game state functions
def get_valid_moves(board):
    column_count = len(board[0])
    return [col for col in range(column_count) if board[0][col] == EMPTY]

def is_terminal_node(board):
    valid_moves = get_valid_moves(board)
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_moves) == 0

def manage_transposition_table():
    global transposition_table
    if len(transposition_table) > MAX_TABLE_SIZE:
        transposition_table.clear()

def winning_move(board, piece):
    row_count = len(board)
    column_count = len(board[0])
    
    # Check horizontal
    for r in range(row_count):
        for c in range(column_count-3):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical
    for c in range(column_count):
        for r in range(row_count-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positive diagonal
    for r in range(row_count-3):
        for c in range(column_count-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negative diagonal
    for r in range(3, row_count):
        for c in range(column_count-3):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    
    return False

def get_next_open_row(board, col):
    row_count = len(board)
    for r in range(row_count-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def drop_piece(board, row, col, piece):
    if row == -1:  # Column is full
        return board
    board_copy = [row[:] for row in board]
    board_copy[row][col] = piece
    return board_copy

def print_board(board, last_move=None):
    print("\nTrạng thái bàn cờ:")
    for row in board:
        print(" | ".join(str(cell) if cell != 0 else "." for cell in row))
    print("-" * (len(board[0]) * 4 - 1))

    if last_move:
        row, col = last_move
        print(f"Quân cờ vừa rơi vào vị trí: Hàng {row+1}, Cột {col+1}")

# Minimax algorithm implementation
def sort_valid_moves_with_boards(valid_moves, board, piece):
    scored_moves = []
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            board_copy = drop_piece(board, row, col, piece)
            score = score_position(board_copy, piece)
            scored_moves.append((col, score, board_copy))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves

def minimax(board, depth, alpha, beta, maximizing_player):
    valid_moves = get_valid_moves(board)
    
    # Convert board to hashable format
    board_tuple = tuple(tuple(row) for row in board)
    state_key = (board_tuple, depth, maximizing_player)

    # Check transposition table
    if state_key in transposition_table:
        return transposition_table[state_key]

    # Check terminal conditions
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                result = (None, 10000000)
            elif winning_move(board, PLAYER_PIECE):
                result = (None, -1000000)
            else:
                result = (None, 0)
        else:
            result = (None, score_position(board, AI_PIECE))
        transposition_table[state_key] = result
        return result
    
    # Maximizing player (AI)    
    if maximizing_player:
        value = -math.inf
        column = None
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, AI_PIECE)
        for col, _, board_copy in sorted_moves:
            new_score = minimax(board_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        transposition_table[state_key] = (column, value)
        return column, value
    
    # Minimizing player (opponent)
    else:
        value = math.inf
        column = None
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, PLAYER_PIECE)
        for col, _, board_copy in sorted_moves:
            new_score = minimax(board_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        transposition_table[state_key] = (column, value)
        return column, value

# API endpoint
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # Get current game state
        board = game_state.board
        valid_moves = game_state.valid_moves
        
        # Update global variables
        global PLAYER_PIECE, AI_PIECE
        AI_PIECE = game_state.current_player
        PLAYER_PIECE = 3 - AI_PIECE
        
        # Manage transposition table
        manage_transposition_table()
        
        # Verify valid moves
        verified_valid_moves = [col for col in valid_moves if get_next_open_row(board, col) != -1]
        
        if not verified_valid_moves:
            raise ValueError("Không có nước đi hợp lệ sau khi xác minh")
        
        # Use minimax algorithm to select the best move
        selected_col, minimax_score = minimax(board, 7, -math.inf, math.inf, True)
        
        # Fallback to random move if needed
        if selected_col is None or selected_col not in verified_valid_moves:
            selected_col = random.choice(verified_valid_moves)
            
        # Make the move
        row = get_next_open_row(board, selected_col)
        if row == -1:
            verified_valid_moves.remove(selected_col)
            if verified_valid_moves:
                selected_col = random.choice(verified_valid_moves)
                row = get_next_open_row(board, selected_col)
            else:
                raise ValueError("Không còn nước đi hợp lệ")
        
        board = drop_piece(board, row, selected_col, AI_PIECE)
        
        # Print the board state
        print_board(board, (row, selected_col))
        
        return AIResponse(move=selected_col)
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        # Fallback strategy if an error occurs
        if game_state.valid_moves:
            col = game_state.valid_moves[0]
            row = get_next_open_row(game_state.board, col)
            if row != -1:
                return AIResponse(move=col)
            
            for col in game_state.valid_moves:
                row = get_next_open_row(game_state.board, col)
                if row != -1:
                    return AIResponse(move=col)
        
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
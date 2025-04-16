from fastapi import FastAPI, HTTPException
import random
import uvicorn
import numpy as np
import math
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

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

# Add a global dictionary for memoization
transposition_table = {}

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

def evaluate_window(window, piece, row_index=None, col_start=None, board=None):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    def is_playable(col_offset):
        if board is None or row_index is None or col_start is None:
            return True  # Không kiểm tra được => vẫn đánh giá
        col = col_start + col_offset
        if row_index == len(board) - 1:  # là hàng cuối
            return True
        return board[row_index + 1][col] != EMPTY  # Ô dưới đã có quân

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

def get_valid_moves(board):
    column_count = len(board[0])
    return [col for col in range(column_count) if board[0][col] == EMPTY]

def score_position(board, piece):
    board_array = np.array(board)
    score = 0
    row_count = len(board)
    column_count = len(board[0])
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    # Score center column - ưu tiên kiểm soát cột giữa
    center_array = [int(i) for i in list(board_array[:, column_count//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(row_count):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(column_count - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece, row_index=r, col_start=c, board=board)

    # Score Vertical
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

    # Thêm đánh giá vị trí theo độ cao - ưu tiên các vị trí gần đáy
    for c in range(column_count):
        for r in range(row_count):
            if board[r][c] == piece:
                # Trọng số tăng dần từ trên xuống dưới bàn cờ
                score += (r + 1) * 0.5

    # Đánh giá mối đe dọa nhiều hướng
    # Kiểm tra các vị trí trống
    for c in range(column_count):
        # Chỉ xét vị trí trống trên cùng của mỗi cột (nơi quân cờ sẽ rơi vào)
        r = get_next_open_row(board, c)
        if r == -1:  # Cột đã đầy
            continue
            
        # Thử đặt quân vào vị trí này
        board_copy = [row[:] for row in board]
        board_copy[r][c] = piece
        
        # Kiểm tra các hướng có thể tạo connect-4
        threat_directions = 0
        
        # Kiểm tra ngang
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(piece) == 3 and row_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra dọc (chỉ kiểm tra hướng xuống vì quân cờ rơi từ trên xuống)
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(piece) == 3 and col_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra đường chéo xuống phải
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra đường chéo xuống trái
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Nếu có nhiều hơn 1 hướng tạo connect-4, đây là mối đe dọa lớn
        if threat_directions > 1:
            score += 100 * threat_directions
        
        # Kiểm tra các mối đe dọa của đối thủ để phòng thủ
        board_copy = [row[:] for row in board]
        board_copy[r][c] = opp_piece
        
        opp_threat_directions = 0
        
        # Kiểm tra tương tự các hướng cho đối thủ
        # Kiểm tra ngang
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(opp_piece) == 3 and row_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra dọc
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(opp_piece) == 3 and col_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra đường chéo xuống phải
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra đường chéo xuống trái
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Ưu tiên phòng thủ cao hơn nếu đối thủ có nhiều mối đe dọa
        if opp_threat_directions > 1:
            score -= 120 * opp_threat_directions  # Điểm trừ nhiều hơn để ưu tiên phòng thủ

    return score

def is_terminal_node(board):
    # Check if game is over
    valid_moves = get_valid_moves(board)
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_moves) == 0

# Hàm kiểm tra và làm sạch transposition table
def manage_transposition_table():
    global transposition_table
    if len(transposition_table) > MAX_TABLE_SIZE:
        # Phương pháp 1: Xóa toàn bộ
        transposition_table.clear()

def winning_move(board, piece):
    # Check horizontal locations
    row_count = len(board)
    column_count = len(board[0])
    
    for r in range(row_count):
        for c in range(column_count-3):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations
    for c in range(column_count):
        for r in range(row_count-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for r in range(row_count-3):
        for c in range(column_count-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
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
    if row == -1:  # Cột đã đầy
        return board  # Trả lại bảng không thay đổi
    board_copy = [row[:] for row in board]  # Create a deep copy
    board_copy[row][col] = piece
    return board_copy

def print_board(board, last_move=None):
    print("\nTrạng thái bàn cờ:")
    for row in board:
        print(" | ".join(str(cell) if cell != 0 else "." for cell in row))
    print("-" * (len(board[0]) * 4 - 1))  # Đường kẻ ngang để phân cách

    # In ra vị trí của quân cờ nếu có
    if last_move:
        row, col = last_move
        print(f"Quân cờ vừa rơi vào vị trí: Hàng {row+1}, Cột {col+1}")  # Cộng thêm 1 để dễ nhìn hơn

def sort_valid_moves_with_boards(valid_moves, board, piece):
    scored_moves = []
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:  # Valid move
            board_copy = drop_piece(board, row, col, piece)
            score = score_position(board_copy, piece)
            scored_moves.append((col, score, board_copy))  # Store column, score, and board state
    # Sort moves by score in descending order
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves

def minimax(board, depth, alpha, beta, maximizing_player):
    # Get current valid moves
    valid_moves = get_valid_moves(board)
    
    # Convert board to a tuple for hashing
    board_tuple = tuple(tuple(row) for row in board)
    state_key = (board_tuple, depth, maximizing_player)

    # Check if the state is already evaluated
    if state_key in transposition_table:
        return transposition_table[state_key]

    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                result = (None, 10000000)
            elif winning_move(board, PLAYER_PIECE):
                result = (None, -1000000)
            else:  # Game is over, no more valid moves
                result = (None, 0)
        else:  # Depth is zero
            result = (None, score_position(board, AI_PIECE))
        transposition_table[state_key] = result
        return result
        
    if maximizing_player:
        value = -math.inf
        column = None
        # Get sorted moves with precomputed board states
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, AI_PIECE)
        for col, _, board_copy in sorted_moves:
            # Use new board state and recursively evaluate
            new_score = minimax(board_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        transposition_table[state_key] = (column, value)
        return column, value
    else:  # Minimizing player
        value = math.inf
        column = None
        # Get sorted moves with precomputed board states
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, PLAYER_PIECE)
        for col, _, board_copy in sorted_moves:
            # Use new board state and recursively evaluate
            new_score = minimax(board_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        transposition_table[state_key] = (column, value)
        return column, value
        
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # Lấy thông tin trạng thái hiện tại
        board = game_state.board
        valid_moves = game_state.valid_moves
        
        # Cập nhật các biến toàn cục theo trạng thái hiện tại
        global PLAYER_PIECE, AI_PIECE
        AI_PIECE = game_state.current_player
        PLAYER_PIECE = 3 - AI_PIECE  # Nếu AI là 1, người chơi là 2; nếu AI là 2, người chơi là 1
        
        # Kiểm tra và làm sạch transposition table
        manage_transposition_table()
        
        # Xác minh lại các nước đi hợp lệ
        verified_valid_moves = [col for col in valid_moves if get_next_open_row(board, col) != -1]
        
        if not verified_valid_moves:
            raise ValueError("Không có nước đi hợp lệ sau khi xác minh")
        
        # Sử dụng thuật toán minimax cho AI
        selected_col, minimax_score = minimax(board, 7, -math.inf, math.inf, True)
        
        if selected_col is None or selected_col not in verified_valid_moves:
            # Fallback to random move if minimax fails
            selected_col = random.choice(verified_valid_moves)
            
        # Thực hiện nước đi
        row = get_next_open_row(board, selected_col)
        if row == -1:  # Kiểm tra thêm lần nữa nếu cột đã đầy
            # Chọn một nước đi khác
            verified_valid_moves.remove(selected_col)
            if verified_valid_moves:
                selected_col = random.choice(verified_valid_moves)
                row = get_next_open_row(board, selected_col)
            else:
                raise ValueError("Không còn nước đi hợp lệ")
        
        board = drop_piece(board, row, selected_col, AI_PIECE)
        
        # In trạng thái bàn cờ sau khi AI chọn nước đi
        print_board(board, (row, selected_col))
        
        return AIResponse(move=selected_col)
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        # Nếu có lỗi nhưng vẫn có nước đi hợp lệ, chọn nước đi đầu tiên
        if game_state.valid_moves:
            # Kiểm tra nước đi đầu tiên có hợp lệ không
            col = game_state.valid_moves[0]
            row = get_next_open_row(game_state.board, col)
            if row != -1:
                return AIResponse(move=col)
            
            # Nếu không hợp lệ, tìm nước đi hợp lệ đầu tiên
            for col in game_state.valid_moves:
                row = get_next_open_row(game_state.board, col)
                if row != -1:
                    return AIResponse(move=col)
        
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
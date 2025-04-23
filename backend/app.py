
import numpy as np
import math
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import traceback

# Constants
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4
MAX_TABLE_SIZE = 1000000
MAX_THINKING_TIME = 2.0  # Thời gian tối đa suy nghĩ (giây)

# Global variables
transposition_table = {}
move_history = {}
opponent_move_history = []
opening_book = {
    # Bàn cờ trống, đánh vào cột giữa
    ((0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0)): 3,
    
    # Người chơi đã đánh vào cột giữa, AI chọn cột bên cạnh
    ((0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 1, 0, 0, 0)): 4,
     
    # Các pattern mở đầu khác có thể thêm vào đây
}

# Pattern definitions for advanced pattern matching
patterns = {
    "trap": [[EMPTY, AI_PIECE, AI_PIECE, EMPTY], [EMPTY, EMPTY, AI_PIECE, AI_PIECE, EMPTY]],
    "block": [[PLAYER_PIECE, PLAYER_PIECE, EMPTY], [EMPTY, PLAYER_PIECE, PLAYER_PIECE]],
    "fork": [[EMPTY, AI_PIECE, EMPTY, AI_PIECE, EMPTY]]
}

# Game state functions
def get_valid_moves(board: List[List[int]]) -> List[int]:
    """Trả về danh sách các cột mà có thể đặt quân vào."""
    column_count = len(board[0])
    return [col for col in range(column_count) if board[0][col] == EMPTY]

def get_next_open_row(board: List[List[int]], col: int) -> int:
    """Tìm hàng trống tiếp theo trong một cột."""
    row_count = len(board)
    for r in range(row_count-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def drop_piece(board: List[List[int]], row: int, col: int, piece: int) -> List[List[int]]:
    """Đặt quân cờ vào bàn cờ và trả về bàn cờ mới."""
    if row == -1:  # Column is full
        return board
    board_copy = [row[:] for row in board]
    board_copy[row][col] = piece
    return board_copy

def winning_move(board: List[List[int]], piece: int) -> bool:
    """Kiểm tra xem người chơi có quân cờ đã thắng chưa."""
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

def is_terminal_node(board: List[List[int]]) -> bool:
    """Kiểm tra xem trạng thái hiện tại có phải là nút cuối cùng không."""
    valid_moves = get_valid_moves(board)
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_moves) == 0

def print_board(board: List[List[int]], last_move: Tuple[int, int] = None) -> None:
    """In bàn cờ ra console để debug."""
    print("\nTrạng thái bàn cờ:")
    for row in board:
        print(" | ".join(str(cell) if cell != 0 else "." for cell in row))
    print("-" * (len(board[0]) * 4 - 1))

    if last_move:
        row, col = last_move
        print(f"Quân cờ vừa rơi vào vị trí: Hàng {row+1}, Cột {col+1}")

# Advanced evaluation functions
def count_threats_in_window(window: List[int], piece: int) -> int:
    """Đếm số mối đe dọa trong một cửa sổ."""
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    # Threat được định nghĩa là 3 quân liền nhau và một ô trống
    if window.count(piece) == 3 and window.count(EMPTY) == 1:
        return 1
    return 0

def evaluate_window(window: List[int], piece: int, row_index: Optional[int] = None, 
                    col_start: Optional[int] = None, board: Optional[List[List[int]]] = None) -> int:
    """Đánh giá một cửa sổ 4 ô."""
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    def is_playable(col_offset):
        """Kiểm tra xem vị trí có thể chơi được không (có ô trống phía dưới)."""
        if board is None or row_index is None or col_start is None:
            return True
        col = col_start + col_offset
        if row_index == len(board) - 1:
            return True
        return board[row_index + 1][col] != EMPTY

    # Các giá trị trọng số điều chỉnh
    win_score = 100000
    three_score = 100
    two_score = 10
    opp_three_score = 80
    opp_two_score = 5
    
    # Trọng số cho các mẫu liên tục và không liên tục
    if window.count(piece) == 4:
        score += win_score
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        empty_index = window.index(EMPTY)
        if is_playable(empty_index):
            score += three_score
            # Thêm điểm cho mẫu liên tục vs không liên tục
            if empty_index > 0 and empty_index < 3:  # Khe trống ở giữa (mẫu không liên tục)
                score += 20
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        # Kiểm tra các mẫu khác nhau của 2 quân cờ và 2 khe trống
        if window == [piece, piece, EMPTY, EMPTY] or window == [EMPTY, EMPTY, piece, piece]:
            score += two_score + 5  # Mẫu liên tục
        elif window == [piece, EMPTY, piece, EMPTY] or window == [EMPTY, piece, EMPTY, piece]:
            score += two_score + 2  # Mẫu rải rác nhưng vẫn tốt
        else:
            score += two_score

    # Phòng thủ - ngăn chặn đối thủ
    if window.count(opp_piece) == 4:
        score -= win_score
    elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        empty_index = window.index(EMPTY)
        if is_playable(empty_index):
            score -= opp_three_score
            # Mẫu liên tục vs không liên tục
            if empty_index > 0 and empty_index < 3:  # Khe trống ở giữa
                score -= 30  # Ưu tiên chặn mẫu sắp thắng cao hơn
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        if window == [opp_piece, opp_piece, EMPTY, EMPTY] or window == [EMPTY, EMPTY, opp_piece, opp_piece]:
            score -= opp_two_score + 5
        else:
            score -= opp_two_score

    return score

def match_pattern(window: List[int], pattern: List[int]) -> bool:
    """Kiểm tra xem một cửa sổ có khớp với mẫu không."""
    if len(window) != len(pattern):
        return False
    for i in range(len(window)):
        if pattern[i] != EMPTY and window[i] != pattern[i]:
            return False
    return True

def score_position(board: List[List[int]], piece: int) -> int:
    """Đánh giá toàn bộ bàn cờ cho một người chơi cụ thể."""
    board_array = np.array(board)
    score = 0
    row_count = len(board)
    column_count = len(board[0])
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    
    # Đánh giá tối ưu hơn - Trọng số cho các vị trí
    position_weights = np.array([
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8, 10, 8, 6, 4],
        [5, 8, 11, 13, 11, 8, 5],
        [5, 8, 11, 13, 11, 8, 5],
        [4, 6, 8, 10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3]
    ])
    
    # Thêm điểm cho vị trí chiến lược
    for r in range(row_count):
        for c in range(column_count):
            if board[r][c] == piece:
                score += position_weights[r][c] * 0.5
            elif board[r][c] == opp_piece:
                score -= position_weights[r][c] * 0.3

    # Score center column - ưu tiên cột giữa
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
    for r in range(3, row_count):
        for c in range(column_count-3):
            window = [board[r-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Evaluate multi-directional threats
    threat_score = evaluate_threats(board, piece)
    score += threat_score
    
    # Đánh giá các mẫu đặc biệt
    pattern_score = evaluate_patterns(board, piece)
    score += pattern_score

    return score

def evaluate_threats(board: List[List[int]], piece: int) -> int:
    """Đánh giá các mối đe dọa đa hướng trên bàn cờ."""
    score = 0
    row_count = len(board)
    column_count = len(board[0])
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    
    for c in range(column_count):
        r = get_next_open_row(board, c)
        if r == -1:  # Cột đã đầy
            continue
            
        # Kiểm tra đặt quân vào vị trí này
        board_copy = drop_piece(board, r, c, piece)
        
        threat_directions = 0
        
        # Kiểm tra mối đe dọa theo chiều ngang
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(piece) == 3 and row_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra mối đe dọa theo chiều dọc
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(piece) == 3 and col_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra mối đe dọa theo đường chéo xuống phải
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Kiểm tra mối đe dọa theo đường chéo xuống trái
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(piece) == 3 and diag_window.count(EMPTY) == 1:
                threat_directions += 1
        
        # Điểm cao cho các mối đe dọa đa chiều
        if threat_directions > 1:
            score += 200 * threat_directions
        
        # Kiểm tra các mối đe dọa của đối thủ tại vị trí này
        board_copy = drop_piece(board, r, c, opp_piece)
        
        opp_threat_directions = 0
        
        # Kiểm tra mối đe dọa ngang của đối thủ
        if c <= column_count - 4:
            row_window = [board_copy[r][c+i] for i in range(WINDOW_LENGTH)]
            if row_window.count(opp_piece) == 3 and row_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra mối đe dọa dọc của đối thủ
        if r <= row_count - 4:
            col_window = [board_copy[r+i][c] for i in range(WINDOW_LENGTH)]
            if col_window.count(opp_piece) == 3 and col_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra mối đe dọa chéo xuống phải của đối thủ
        if c <= column_count - 4 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c+i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Kiểm tra mối đe dọa chéo xuống trái của đối thủ
        if c >= 3 and r <= row_count - 4:
            diag_window = [board_copy[r+i][c-i] for i in range(WINDOW_LENGTH)]
            if diag_window.count(opp_piece) == 3 and diag_window.count(EMPTY) == 1:
                opp_threat_directions += 1
        
        # Ưu tiên phòng thủ nếu đối thủ có nhiều mối đe dọa
        if opp_threat_directions > 1:
            score -= 250 * opp_threat_directions

    return score

def evaluate_patterns(board: List[List[int]], piece: int) -> int:
    """Đánh giá các mẫu đặc biệt trên bàn cờ."""
    score = 0
    row_count = len(board)
    column_count = len(board[0])
    
    # Kiểm tra các mẫu "trap" (bẫy)
    for r in range(row_count):
        for c in range(column_count - 4):
            window = [board[r][c+i] for i in range(5)]
            if window.count(piece) == 2 and window.count(EMPTY) == 3:
                if (window[0] == piece and window[4] == piece) or \
                   (window[1] == piece and window[3] == piece):
                    score += 50  # Mẫu bẫy tiềm năng
    
    # Kiểm tra các mẫu "fork" (ngã ba)
    for c in range(column_count):
        r = get_next_open_row(board, c)
        if r == -1:
            continue
        
        temp_board = [row[:] for row in board]
        temp_board[r][c] = piece
        
        # Đếm số hướng đe dọa sau khi đặt quân
        threats = 0
        
        # Kiểm tra các hướng
        directions = [
            [(0, 1), (0, -1)],  # Ngang
            [(1, 0), (-1, 0)],  # Dọc
            [(1, 1), (-1, -1)],  # Chéo xuống phải
            [(1, -1), (-1, 1)]   # Chéo xuống trái
        ]
        
        for dir_pair in directions:
            count = 1  # Đếm quân cờ hiện tại
            empty_spots = 0
            
            for dr, dc in dir_pair:
                for i in range(1, 4):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < row_count and 0 <= nc < column_count:
                        if temp_board[nr][nc] == piece:
                            count += 1
                        elif temp_board[nr][nc] == EMPTY:
                            empty_spots += 1
                            break
                        else:
                            break
                    else:
                        break
            
            if count == 2 and empty_spots >= 1:
                threats += 1
            elif count == 3 and empty_spots >= 1:
                threats += 2
        
        if threats >= 2:
            score += 150 * threats  # Nhiều mối đe dọa tiềm năng
    
    return score

# Chức năng phát hiện tình huống đặc biệt
def check_immediate_win(board: List[List[int]], piece: int) -> Optional[int]:
    """Kiểm tra và trả về nước đi chiến thắng ngay lập tức nếu có."""
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            temp_board = drop_piece(board, row, col, piece)
            if winning_move(temp_board, piece):
                return col
    return None

def check_immediate_threat(board: List[List[int]], opponent_piece: int) -> Optional[int]:
    """Kiểm tra và trả về nước đi để ngăn chặn chiến thắng của đối thủ nếu có."""
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            temp_board = drop_piece(board, row, col, opponent_piece)
            if winning_move(temp_board, opponent_piece):
                return col
    return None

def check_trap_move(board: List[List[int]], piece: int) -> Optional[int]:
    """Kiểm tra nước đi tạo bẫy (tạo hai đường thắng tiềm năng)."""
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row == -1:
            continue
            
        # Đặt quân thử
        temp_board = drop_piece(board, row, col, piece)
        
        # Đếm số hướng có thể thắng
        winning_paths = 0
        
        # Kiểm tra từng cột sau khi đặt quân
        for next_col in get_valid_moves(temp_board):
            next_row = get_next_open_row(temp_board, next_col)
            if next_row == -1:
                continue
                
            next_board = drop_piece(temp_board, next_row, next_col, piece)
            if winning_move(next_board, piece):
                winning_paths += 1
        
        if winning_paths >= 2:
            return col
    
    return None

def get_move_from_opening_book(board: List[List[int]]) -> Optional[int]:
    """Lấy nước đi từ opening book nếu có."""
    board_tuple = tuple(tuple(row) for row in board)
    if board_tuple in opening_book:
        return opening_book[board_tuple]
    return None

def get_search_depth(board: List[List[int]], valid_moves: List[int]) -> int:
    """Điều chỉnh độ sâu tìm kiếm dựa trên giai đoạn trò chơi."""
    pieces_count = sum(row.count(PLAYER_PIECE) + row.count(AI_PIECE) for row in board)
    total_positions = len(board) * len(board[0])
    
    # Độ sâu tùy thuộc vào số lượng quân cờ trên bàn
    if pieces_count < total_positions * 0.3:
        return 5  # Giai đoạn đầu
    elif pieces_count < total_positions * 0.7:
        return 6  # Giai đoạn giữa
    else:
        return 8  # Giai đoạn cuối (có thể rất sâu vì ít nước đi hơn)

def manage_transposition_table() -> None:
    """Quản lý bảng transposition để tránh sử dụng quá nhiều bộ nhớ."""
    global transposition_table
    if len(transposition_table) > MAX_TABLE_SIZE:
        # Xóa 30% các mục ít được truy cập nhất
        items_to_keep = int(MAX_TABLE_SIZE * 0.7)
        # Sắp xếp theo số lần truy cập (nếu có)
        sorted_items = sorted(
            transposition_table.items(), 
            key=lambda x: x[1][2] if len(x[1]) > 2 else 0, 
            reverse=True
        )
        transposition_table = dict(sorted_items[:items_to_keep])

def update_history(move: int, depth: int) -> None:
    """Cập nhật history heuristic cho việc sắp xếp nước đi."""
    global move_history
    if move not in move_history:
        move_history[move] = 0
    move_history[move] += 2**depth  # Nước đi tốt ở độ sâu lớn nhận điểm cao hơn

def sort_valid_moves_with_boards(valid_moves: List[int], board: List[List[int]], piece: int) -> List[Tuple[int, int, List[List[int]]]]:
    """Sắp xếp các nước đi hợp lệ theo đánh giá sơ bộ."""
    scored_moves = []
    
    # Sử dụng history heuristic để ưu tiên các nước đi tốt từ trước
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            board_copy = drop_piece(board, row, col, piece)
            
            # Tính điểm nhanh
            quick_score = 0
            
            # Cộng điểm từ history heuristic
            if col in move_history:
                quick_score += move_history[col] * 0.001
                
            # Ưu tiên cột giữa và các cột gần giữa
            column_count = len(board[0])
            center_distance = abs(col - column_count // 2)
            quick_score += (column_count // 2 - center_distance) * 10
            
            # Đánh giá nhanh vị trí này
            score = quick_score + score_position(board_copy, piece) * 0.2
            
            scored_moves.append((col, score, board_copy))
    
    # Sắp xếp theo điểm số giảm dần
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves

def analyze_opponent_moves(opponent_moves: List[int], board_size: Tuple[int, int]) -> Dict[str, Any]:
    """Phân tích xu hướng trong các nước đi của đối thủ."""
    analysis = {
        "prefers_center": False,
        "prefers_sides": False,
        "aggressive": False,
        "defensive": False
    }
    
    # Không đủ dữ liệu để phân tích
    if len(opponent_moves) < 3:
        return analysis
        
    column_count = board_size[1]
    center_column = column_count // 2
    center_moves = sum(1 for move in opponent_moves if move == center_column)
    side_moves = sum(1 for move in opponent_moves if move == 0 or move == column_count - 1)
    
    # Phân tích sở thích cột
    if center_moves / len(opponent_moves) > 0.4:
        analysis["prefers_center"] = True
    if side_moves / len(opponent_moves) > 0.3:
        analysis["prefers_sides"] = True
        
    # Phân tích tính hung hăng/phòng thủ có thể được thêm vào sau
    
    return analysis

def iterative_deepening(board: List[List[int]], max_depth: int, time_limit: float) -> Tuple[Optional[int], int, int]:
    """Thực hiện tìm kiếm iterative deepening với giới hạn thời gian."""
    best_move = None
    best_score = -math.inf
    positions_evaluated = 0
    
    start_time = time.time()
    
    # Bắt đầu từ độ sâu 1 và tăng dần
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit * 0.8:  # Để lại 20% thời gian để tiến hành nước đi
            break
            
        move, score, positions = minimax_with_time(board, depth, -math.inf, math.inf, True, start_time, time_limit)
        
        # Cập nhật transposition table count
        positions_evaluated += positions
        
        if move is not None:
            best_move = move
            best_score = score
            
        # Nếu tìm thấy nước thắng, không cần tìm sâu hơn
        if score > 99000:
            break
    
    return best_move, best_score, positions_evaluated

def minimax_with_time(board: List[List[int]], depth: int, alpha: float, beta: float, 
                     maximizing_player: bool, start_time: float, time_limit: float) -> Tuple[Optional[int], float, int]:
    """Thuật toán minimax với cắt tỉa alpha-beta và giới hạn thời gian."""
    # Kiểm tra giới hạn thời gian
    if time.time() - start_time > time_limit:
        return None, 0, 0

    valid_moves = get_valid_moves(board)
    positions_evaluated = 1
    
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
                result = (None, 10000000, 1)
            elif winning_move(board, PLAYER_PIECE):
                result = (None, -1000000, 1)
            else:
                result = (None, 0, 1)
        else:
            result = (None, score_position(board, AI_PIECE), 1)
        transposition_table[state_key] = result
        return result
    
    # Maximizing player (AI)    
    if maximizing_player:
        value = -math.inf
        column = None
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, AI_PIECE)
        for col, _, board_copy in sorted_moves:
            # Kiểm tra thời gian trước mỗi lần gọi đệ quy
            if time.time() - start_time > time_limit:
                break
                
            new_score, _, positions = minimax_with_time(board_copy, depth-1, alpha, beta, False, start_time, time_limit)
            positions_evaluated += positions
            
            if new_score is None:  # Hết thời gian
                continue
                
            if new_score > value:
                value = new_score
                column = col
                
            alpha = max(alpha, value)
            
            # Cập nhật history heuristic
            if depth == 1 and column is not None:
                update_history(column, depth)
                
            if alpha >= beta:
                break
                
        result = (column, value, positions_evaluated)
        transposition_table[state_key] = result
        return result
    
    # Minimizing player (opponent)
    else:
        value = math.inf
        column = None
        sorted_moves = sort_valid_moves_with_boards(valid_moves, board, PLAYER_PIECE)
        for col, _, board_copy in sorted_moves:
            # Kiểm tra thời gian trước mỗi lần gọi đệ quy
            if time.time() - start_time > time_limit:
                break
                
            new_score, _, positions = minimax_with_time(board_copy, depth-1, alpha, beta, True, start_time, time_limit)
            positions_evaluated += positions
            
            if new_score is None:  # Hết thời gian
                continue
                
            if new_score < value:
                value = new_score
                column = col
                
            beta = min(beta, value)
            
            if alpha >= beta:
                break
                
        result = (column, value, positions_evaluated)
        transposition_table[state_key] = result
        return result

def find_best_move(board: List[List[int]], time_limit: float = MAX_THINKING_TIME) -> Dict[str, Any]:
    """Tìm nước đi tốt nhất cho AI với các chiến lược cải tiến."""
    global transposition_table, move_history, opponent_move_history
    
    start_time = time.time()
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        return {"move": None, "evaluation": 0, "positions_evaluated": 0, "thinking_time_ms": 0}
    
    # Quản lý bảng transposition
    manage_transposition_table()
    
    # Kiểm tra opening book
    book_move = get_move_from_opening_book(board)
    if book_move is not None and book_move in valid_moves:
        print("Sử dụng nước đi từ opening book:", book_move)
        return {
            "move": book_move,
            "evaluation": 0,
            "positions_evaluated": 0,
            "thinking_time_ms": int((time.time() - start_time) * 1000),
            "source": "opening_book"
        }
    
    # Kiểm tra nước thắng ngay lập tức
    win_move = check_immediate_win(board, AI_PIECE)
    if win_move is not None:
        print("Phát hiện nước thắng ngay lập tức:", win_move)
        return {
            "move": win_move,
            "evaluation": 100000,
            "positions_evaluated": len(valid_moves),
            "thinking_time_ms": int((time.time() - start_time) * 1000),
            "source": "immediate_win"
        }
    
    # Kiểm tra và chặn nước thắng của đối thủ
    block_move = check_immediate_threat(board, PLAYER_PIECE)
    if block_move is not None:
        print("Phát hiện và chặn nước thắng của đối thủ:", block_move)
        return {
            "move": block_move,
            "evaluation": 50000,
            "positions_evaluated": len(valid_moves) * 2,
            "thinking_time_ms": int((time.time() - start_time) * 1000),
            "source": "block_threat"
        }
    
    # Kiểm tra nước đi bẫy (tạo hai mối đe dọa cùng lúc)
    trap_move = check_trap_move(board, AI_PIECE)
    if trap_move is not None:
        print("Phát hiện nước đi bẫy:", trap_move)
        return {
            "move": trap_move,
            "evaluation": 80000,
            "positions_evaluated": len(valid_moves) * 5,
            "thinking_time_ms": int((time.time() - start_time) * 1000),
            "source": "trap_move"
        }
    
    # Phân tích xu hướng đối thủ
    if len(opponent_move_history) >= 3:
        opponent_analysis = analyze_opponent_moves(opponent_move_history, (len(board), len(board[0])))
        # Có thể điều chỉnh chiến lược dựa trên phân tích
    
    # Xác định độ sâu tìm kiếm dựa trên giai đoạn trò chơi
    search_depth = get_search_depth(board, valid_moves)
    
    # Sử dụng iterative deepening với giới hạn thời gian
    remaining_time = time_limit - (time.time() - start_time)
    selected_col, minimax_score, positions_evaluated = iterative_deepening(board, search_depth, remaining_time)
    
    # Fallback nếu không tìm được nước đi
    if selected_col is None:
        # Sắp xếp các nước đi theo đánh giá trước
        scored_moves = []
        for col in valid_moves:
            row = get_next_open_row(board, col)
            if row != -1:
                board_copy = drop_piece(board, row, col, AI_PIECE)
                score = score_position(board_copy, AI_PIECE)
                scored_moves.append((col, score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        if scored_moves:
            selected_col = scored_moves[0][0]
        else:
            selected_col = random.choice(valid_moves)
    
    # Cập nhật lịch sử nước đi
    update_history(selected_col, 5)  # Giả định độ sâu tìm kiếm cao
    
    return {
        "move": selected_col,
        "evaluation": minimax_score,
        "positions_evaluated": positions_evaluated,
        "thinking_time_ms": int((time.time() - start_time) * 1000),
        "search_depth": search_depth,
        "source": "minimax"
    }

def register_opponent_move(col: int) -> None:
    """Ghi lại nước đi của đối thủ để phân tích."""
    global opponent_move_history
    opponent_move_history.append(col)
    if len(opponent_move_history) > 20:  # Chỉ lưu trữ 20 nước đi gần nhất
        opponent_move_history.pop(0)


# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Connect4 AI API", 
              description="API thông minh cho trò chơi Connect 4 với thuật toán Minimax cải tiến",
              version="2.0.0")

# Bật CORS để cho phép gọi từ các domain khác
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool
    time_limit_ms: Optional[int] = None  # Cho phép chỉ định giới hạn thời gian

class AIResponse(BaseModel):
    move: int
    analytics: Optional[Dict[str, Any]] = None

# Route để gọi API xác định nước đi
@app.post("/api/connect4-move", response_model=AIResponse)
async def make_move(game_state: GameState) -> AIResponse:
    """
    API trả về nước đi tốt nhất cho AI dựa trên trạng thái hiện tại của trò chơi.
    
    - **board**: Ma trận biểu diễn bàn cờ, 0=trống, 1=người chơi, 2=AI
    - **current_player**: Người chơi hiện tại (1 hoặc 2)
    - **valid_moves**: Danh sách các cột mà có thể đặt quân vào
    - **is_new_game**: Boolean cho biết đây có phải là ván mới không
    - **time_limit_ms**: (Tùy chọn) Giới hạn thời gian suy nghĩ tính bằng millisecond
    """
    try:
        # Bắt đầu tính thời gian phản hồi
        start_time = time.time()
        
        # Xác minh đầu vào
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # Lấy trạng thái hiện tại
        board = game_state.board
        valid_moves = game_state.valid_moves
        
        # Xử lý khi bắt đầu ván mới
        if game_state.is_new_game:
            # Reset transposition table và lịch sử nước đi
            global transposition_table, opponent_move_history
            transposition_table = {}
            opponent_move_history = []
            print("Bắt đầu ván mới - Đã reset trạng thái")
        
        # Cập nhật AI_PIECE và PLAYER_PIECE dựa trên người chơi hiện tại
        global AI_PIECE, PLAYER_PIECE
        AI_PIECE = game_state.current_player
        PLAYER_PIECE = 3 - AI_PIECE
        
        # Xác minh lại các nước đi hợp lệ
        verified_valid_moves = [col for col in valid_moves if get_next_open_row(board, col) != -1]
        
        if not verified_valid_moves:
            raise ValueError("Không có nước đi hợp lệ sau khi xác minh")
        
        # Xác định giới hạn thời gian (mặc định hoặc từ request)
        time_limit = MAX_THINKING_TIME
        if game_state.time_limit_ms is not None:
            time_limit = max(0.2, min(5.0, game_state.time_limit_ms / 1000))  # Giới hạn từ 0.2s đến 5.0s
        
        # Gọi thuật toán tìm nước đi tốt nhất
        result = find_best_move(board, time_limit)
        selected_col = result["move"]
        
        # Kiểm tra lại nước đi có hợp lệ không
        if selected_col is None or selected_col not in verified_valid_moves:
            print("Nước đi được chọn không hợp lệ, chọn nước đi ngẫu nhiên...")
            import random
            selected_col = random.choice(verified_valid_moves)
            result["move"] = selected_col
            result["source"] = "fallback_random"
        
        # Tạo bảng mới với nước đi và in ra console
        row = get_next_open_row(board, selected_col)
        if row != -1:
            new_board = drop_piece(board, row, selected_col, AI_PIECE)
            print_board(new_board, (row, selected_col))
            
            # Kiểm tra kết quả
            if winning_move(new_board, AI_PIECE):
                print("🎉 AI thắng!")
                result["game_result"] = "ai_win"
            elif len(get_valid_moves(new_board)) == 0:
                print("🤝 Ván cờ hòa!")
                result["game_result"] = "draw"
        
        # Tính toán thời gian phản hồi
        response_time = time.time() - start_time
        print(f"Thời gian phản hồi: {response_time:.3f}s")
        
        # Thêm thông tin analytics
        analytics = {
            "move": selected_col,
            "evaluation": result.get("evaluation", 0),
            "positions_evaluated": result.get("positions_evaluated", 0),
            "thinking_time_ms": result.get("thinking_time_ms", int(response_time * 1000)),
            "source": result.get("source", "unknown"),
            "response_time_ms": int(response_time * 1000)
        }
        
        if "search_depth" in result:
            analytics["search_depth"] = result["search_depth"]
        
        if "game_result" in result:
            analytics["game_result"] = result["game_result"]
        
        return AIResponse(move=selected_col, analytics=analytics)
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        print(traceback.format_exc())
        
        # Fallback strategy trong trường hợp có lỗi
        if game_state.valid_moves:
            for col in game_state.valid_moves:
                row = get_next_open_row(game_state.board, col)
                if row != -1:
                    return AIResponse(
                        move=col, 
                        analytics={"error": str(e), "source": "error_fallback"}
                    )
        
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/register-opponent-move")
async def register_move(col: int):
    """
    Ghi lại nước đi của người chơi để phân tích xu hướng.
    """
    try:
        register_opponent_move(col)
        return {"status": "success", "message": f"Đã ghi lại nước đi tại cột {col}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/test")
async def health_check():
    """Kiểm tra trạng thái hoạt động của API."""
    return {
        "status": "ok", 
        "message": "Server đang hoạt động", 
        "version": "2.0.0",
        "positions_in_memory": len(transposition_table)
    }

@app.get("/api/stats")
async def get_stats():
    """Trả về các thống kê hiện tại của AI."""
    return {
        "transposition_table_size": len(transposition_table),
        "opponent_moves_analyzed": len(opponent_move_history),
        "memory_usage_mb": get_memory_usage()
    }

def get_memory_usage():
    """Ước tính lượng bộ nhớ đang sử dụng (MB)."""
    try:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Chuyển từ byte sang MB
    except ImportError:
        return -1  # Không có thư viện psutil

# Chạy ứng dụng
if __name__ == "__main__":
    print("Khởi động Connect 4 AI API...")
    print(f"Giới hạn thời gian mặc định: {MAX_THINKING_TIME}s")
    uvicorn.run(app, host="0.0.0.0", port=8080)
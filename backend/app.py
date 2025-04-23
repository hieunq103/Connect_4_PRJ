
import numpy as np
import math
import time
import random
from typing import List, Tuple, Dict, Any, Optional

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
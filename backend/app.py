from fastapi import FastAPI, HTTPException
import random
import uvicorn
import numpy as np
import math
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import random
import time

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

# Phần Zobrist hashing - thêm vào sau các hằng số
class ZobristHash:
    def __init__(self, rows, cols, pieces):
        self.rows = rows
        self.cols = cols
        self.pieces = pieces
        self.zobrist_table = [[[random.getrandbits(64) for _ in range(pieces)] 
                               for _ in range(cols)] for _ in range(rows)]
        
    def hash_board(self, board):
        h = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] != EMPTY:
                    piece_idx = board[r][c] - 1  # Chuyển 1/2 thành 0/1
                    h ^= self.zobrist_table[r][c][piece_idx]
        return h

# Cấu trúc mới cho transposition table
class TranspositionTable:
    def __init__(self, max_size=MAX_TABLE_SIZE):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def store(self, hash_key, depth, value, move, flag):
        """
        Lưu kết quả vào bảng.
        flag: 0 = exact, 1 = lower bound, 2 = upper bound
        """
        if len(self.table) >= self.max_size:
            # Replacement scheme: thay thế mục có độ sâu thấp hơn
            for k in list(self.table.keys())[:1000]:  # Chỉ kiểm tra 1000 mục đầu tiên để tiết kiệm thời gian
                if self.table[k]['depth'] < depth:
                    del self.table[k]
                    break
            else:  # Nếu không tìm thấy mục nào để thay thế
                return
                
        entry = {
            'value': value,
            'depth': depth,
            'move': move,
            'flag': flag,
            'time': time.time()  # Thêm timestamp để có thể loại bỏ các mục cũ
        }
        self.table[hash_key] = entry
    
    def lookup(self, hash_key, depth, alpha, beta):
        """
        Tìm kiếm trạng thái trong bảng.
        Trả về (hit, value, move)
        """
        if hash_key in self.table:
            entry = self.table[hash_key]
            if entry['depth'] >= depth:
                if entry['flag'] == 0:  # Exact value
                    self.hits += 1
                    return True, entry['value'], entry['move']
                elif entry['flag'] == 1 and entry['value'] >= beta:  # Lower bound
                    self.hits += 1
                    return True, entry['value'], entry['move']
                elif entry['flag'] == 2 and entry['value'] <= alpha:  # Upper bound
                    self.hits += 1
                    return True, entry['value'], entry['move']
            self.collisions += 1
        self.misses += 1
        return False, 0, None
    
    def clear(self):
        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def size(self):
        return len(self.table)
    
    def stats(self):
        return {
            'size': len(self.table),
            'hits': self.hits,
            'misses': self.misses,
            'collisions': self.collisions,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

# Thay thế biến transposition_table toàn cục
zobrist_hasher = None
transposition_table = None

# Pydantic models
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

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
    score += center_count * 4

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

def is_unstable_position(board):
    # Kiểm tra có đe dọa thắng của một trong hai bên
    for piece in [PLAYER_PIECE, AI_PIECE]:
        for col in range(len(board[0])):
            row = get_next_open_row(board, col)
            if row != -1:
                board_copy = drop_piece(board, row, col, piece)
                if winning_move(board_copy, piece):
                    return True
    
    # Kiểm tra có đe dọa tạo 3 quân liên tiếp
    row_count = len(board)
    col_count = len(board[0])
    
    for r in range(row_count):
        for c in range(col_count - 2):
            # Kiểm tra ngang
            window = [board[r][c+i] for i in range(3)]
            for piece in [PLAYER_PIECE, AI_PIECE]:
                if window.count(piece) == 2 and window.count(EMPTY) == 1:
                    return True
    
    # Tương tự cho dọc và chéo...
    
    return False

# Tìm kiếm quiescence
def quiescence_search(board, alpha, beta, maximizing_player, depth=3):
    if depth == 0:
        return score_position(board, AI_PIECE)
    
    is_terminal = is_terminal_node(board)
    if is_terminal:
        if winning_move(board, AI_PIECE):
            return 10000000
        elif winning_move(board, PLAYER_PIECE):
            return -1000000
        else:
            return 0
    
    # Chỉ xem xét các nước đánh làm thay đổi trạng thái đáng kể
    quiet_score = score_position(board, AI_PIECE)
    
    if maximizing_player:
        if quiet_score >= beta:
            return beta
        alpha = max(alpha, quiet_score)
        
        # Chỉ kiểm tra các nước "không yên tĩnh"
        valid_moves = get_valid_moves(board)
        forcing_moves = []
        
        for col in valid_moves:
            row = get_next_open_row(board, col)
            if row != -1:
                board_copy = drop_piece(board, row, col, AI_PIECE)
                # Chỉ xem xét các nước tạo đe dọa
                if creates_threat(board_copy, AI_PIECE, row, col):
                    forcing_moves.append((col, board_copy))
        
        # Nếu không có nước đe dọa, trả về đánh giá tĩnh
        if not forcing_moves:
            return quiet_score
            
        for col, board_copy in forcing_moves:
            value = quiescence_search(board_copy, alpha, beta, False, depth - 1)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return alpha
    else:
        if quiet_score <= alpha:
            return alpha
        beta = min(beta, quiet_score)
        
        valid_moves = get_valid_moves(board)
        forcing_moves = []
        
        for col in valid_moves:
            row = get_next_open_row(board, col)
            if row != -1:
                board_copy = drop_piece(board, row, col, PLAYER_PIECE)
                if creates_threat(board_copy, PLAYER_PIECE, row, col):
                    forcing_moves.append((col, board_copy))
        
        if not forcing_moves:
            return quiet_score
            
        for col, board_copy in forcing_moves:
            value = quiescence_search(board_copy, alpha, beta, True, depth - 1)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return beta

def creates_threat(board, piece, row, col):
    """
    Kiểm tra nếu nước đi tạo ra mối đe dọa (3 quân liên tiếp với 1 ô trống).
    """
    row_count = len(board)
    col_count = len(board[0])

    # Kiểm tra hàng ngang
    for delta in range(-3, 1):
        if 0 <= col + delta and col + delta + 3 < col_count:
            window = [board[row][col + delta + i] for i in range(4)]
            if window.count(piece) == 3 and window.count(EMPTY) == 1:
                return True

    # Kiểm tra cột dọc
    if row <= row_count - 4:
        window = [board[row + i][col] for i in range(4)]
        if window.count(piece) == 3 and window.count(EMPTY) == 1:
            return True

    # Kiểm tra đường chéo chính (từ trên trái xuống dưới phải)
    for delta in range(-3, 1):
        if 0 <= row + delta and row + delta + 3 < row_count and 0 <= col + delta and col + delta + 3 < col_count:
            window = [board[row + delta + i][col + delta + i] for i in range(4)]
            if window.count(piece) == 3 and window.count(EMPTY) == 1:
                return True

    # Kiểm tra đường chéo phụ (từ trên phải xuống dưới trái)
    for delta in range(-3, 1):
        if 0 <= row - delta - 3 and row - delta < row_count and 0 <= col + delta and col + delta + 3 < col_count:
            window = [board[row - delta - i][col + delta + i] for i in range(4)]
            if window.count(piece) == 3 and window.count(EMPTY) == 1:
                return True

    return False

# Kiểm tra xem có đang bị chiếu (trong nguy hiểm) không
def in_check(board, piece):
    opponent = 3 - piece
    # Kiểm tra xem đối thủ có thể thắng trong nước tiếp theo không
    valid_moves = get_valid_moves(board)
    
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            board_copy = drop_piece(board, row, col, opponent)
            if winning_move(board_copy, opponent):
                return True
    
    return False

def find_best_move(board, max_depth=7, time_limit=5.0):
    start_time = time.time()
    best_move = None
    global transposition_table
    
    # Reset hit statistics
    if transposition_table:
        transposition_table.clear()
    
    # Dictionary lưu trữ lịch sử nước đi thành công
    history_table = {}
    # List lưu trữ killer moves cho mỗi độ sâu
    killer_moves = [None] * (max_depth + 1)
    
    try:
        for depth in range(1, max_depth + 1):
            # Kiểm tra nếu còn thời gian
            if time.time() - start_time > time_limit * 0.8:  # Sử dụng 80% thời gian giới hạn
                print(f"Đã hết thời gian ở độ sâu {depth-1}")
                break
                
            move, score = minimax_with_enhancements(board, depth, -math.inf, math.inf, True, 
                                                  start_time, time_limit, history_table, killer_moves)
            
            if move is not None:
                best_move = move
            
            # In tiến độ
            print(f"Độ sâu {depth}, nước đi tốt nhất: {best_move}, điểm: {score}")
            
            # Cắt ngắn nếu tìm thấy nước thắng hoặc thua chắc chắn
            if score >= 9000000:  # Nước thắng
                print("Đã tìm thấy nước thắng!")
                break
            elif score <= -9000000:  # Nước thua không thể tránh
                print("Đã tìm thấy nước phòng thủ tốt nhất!")
                break
                
    except TimeoutError:
        print("Hết thời gian tìm kiếm")
    
    # Nếu không tìm được nước đi tốt nhất, chọn nước đi có điểm số cao nhất
    if best_move is None:
        valid_moves = get_valid_moves(board)
        best_move = max(valid_moves, key=lambda col: score_position(drop_piece(board, get_next_open_row(board, col), col, AI_PIECE)))
    
    # In thống kê transposition table
    if transposition_table:
        stats = transposition_table.stats()
        print(f"Transposition table stats: size={stats['size']}, hits={stats['hits']}, hit_rate={stats['hit_rate']:.2f}")
        
    return best_move

def is_terminal_node(board):
    valid_moves = get_valid_moves(board)
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_moves) == 0

def initialize_tables(rows=6, cols=7):
    global zobrist_hasher, transposition_table
    if zobrist_hasher is None or transposition_table is None:
        zobrist_hasher = ZobristHash(rows, cols, 2)  # 2 loại quân (PLAYER_PIECE và AI_PIECE)
        transposition_table = TranspositionTable(MAX_TABLE_SIZE)
    else:
        print("Các bảng đã được khởi tạo, không cần khởi tạo lại.")

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

def sort_moves_enhanced(valid_moves, board, piece, depth, history_table, killer_moves):
    scored_moves = []
    
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:
            # Tính điểm cơ bản
            board_copy = drop_piece(board, row, col, piece)
            score = score_position(board_copy, piece)
            
            # Ưu tiên 1: Kiểm tra nước thắng ngay lập tức
            if winning_move(board_copy, piece):
                score += 100000
            
            # Ưu tiên 2: Kiểm tra nước phòng thủ (đối thủ sắp thắng)
            opponent_piece = 3 - piece
            for opp_col in valid_moves:
                opp_row = get_next_open_row(board, opp_col)
                if opp_row != -1:
                    opp_board = drop_piece(board, opp_row, opp_col, opponent_piece)
                    if winning_move(opp_board, opponent_piece):
                        if col == opp_col:  # Nếu nước phòng thủ trùng với nước thắng
                            score += 90000
                        else:
                            score += 80000
            
            # Ưu tiên 3: Killer move tại độ sâu này
            if killer_moves[depth] == col:
                score += 50000
            
            # Ưu tiên 4: History heuristic
            history_score = history_table.get((col, depth), 0)
            score += history_score
            
            scored_moves.append((col, score, board_copy))
    
    # Sắp xếp theo điểm giảm dần
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves

def minimax_with_enhancements(board, depth, alpha, beta, maximizing_player, 
                             start_time, time_limit, history_table, killer_moves):
    # Kiểm tra thời gian
    if time.time() - start_time > time_limit:
        raise TimeoutError("Hết thời gian tìm kiếm")
    
    valid_moves = get_valid_moves(board)
    hash_key = zobrist_hasher.hash_board(board)
    
    # Kiểm tra transposition table
    hit, cached_value, cached_move = transposition_table.lookup(hash_key, depth, alpha, beta)
    if hit:
        return cached_move, cached_value
    
    # Kiểm tra điều kiện kết thúc
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 10000000
            elif winning_move(board, PLAYER_PIECE):
                return None, -1000000
            else:
                return None, 0
        else:
            # Thêm quiescence search cho các vị trí không ổn định
            if is_unstable_position(board):
                q_score = quiescence_search(board, alpha, beta, maximizing_player, 3)  # Độ sâu quiescence tối đa là 3
                return None, q_score
            return None, score_position(board, AI_PIECE)
    
    # Sắp xếp nước đi bằng các heuristic nâng cao
    sorted_moves = sort_moves_enhanced(valid_moves, board, 
                                      AI_PIECE if maximizing_player else PLAYER_PIECE,
                                      depth, history_table, killer_moves)
    
    # Maximizing player (AI)
    if maximizing_player:
        value = -math.inf
        column = valid_moves[0] if valid_moves else None
        
        for i, (col, _, board_copy) in enumerate(sorted_moves):
            # Late Move Reduction: giảm độ sâu cho các nước đi kém hứa hẹn
            reduction = 0
            if depth >= 3 and i >= 2 and not is_terminal_node(board_copy):
                reduction = 1
                
            # Null Move Pruning khi vị trí đủ tốt và không trong zugzwang
            if depth >= 3 and not in_check(board, AI_PIECE) and i == 0:
                null_board = [row[:] for row in board]  # Giả vờ không đi
                null_value = -minimax_with_enhancements(null_board, depth-3, -beta, -beta+1, 
                                                     False, start_time, time_limit, 
                                                     history_table, killer_moves)[1]
                if null_value >= beta:
                    # Xác nhận lại với tìm kiếm thực
                    verify_value = minimax_with_enhancements(board_copy, depth-1, alpha, beta, 
                                                          False, start_time, time_limit, 
                                                          history_table, killer_moves)[1]
                    if verify_value >= beta:
                        transposition_table.store(hash_key, depth, beta, col, 1)  # lower bound
                        return col, beta
            
            new_score = minimax_with_enhancements(board_copy, depth-1-reduction, alpha, beta, 
                                               False, start_time, time_limit, 
                                               history_table, killer_moves)[1]
            
            # Nếu áp dụng LMR và kết quả hứa hẹn, tìm kiếm lại với độ sâu đầy đủ
            if reduction > 0 and new_score > alpha:
                new_score = minimax_with_enhancements(board_copy, depth-1, alpha, beta, 
                                                   False, start_time, time_limit, 
                                                   history_table, killer_moves)[1]
            
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                # Cập nhật killer moves và history table
                killer_moves[depth] = col
                history_table[(col, depth)] = history_table.get((col, depth), 0) + 2**depth
                
                # Lưu cận dưới (lower bound)
                transposition_table.store(hash_key, depth, value, column, 1)
                break
                
        # Lưu giá trị chính xác
        transposition_table.store(hash_key, depth, value, column, 0)
        return column, value
    
    # Minimizing player (người chơi)
    else:
        value = math.inf
        column = valid_moves[0] if valid_moves else None
        
        for i, (col, _, board_copy) in enumerate(sorted_moves):
            # Late Move Reduction cho nước đi kém hứa hẹn
            reduction = 0
            if depth >= 3 and i >= 2 and not is_terminal_node(board_copy):
                reduction = 1
                
            new_score = minimax_with_enhancements(board_copy, depth-1-reduction, alpha, beta, 
                                               True, start_time, time_limit, 
                                               history_table, killer_moves)[1]
            
            # Nếu áp dụng LMR và kết quả hứa hẹn, tìm kiếm lại với độ sâu đầy đủ
            if reduction > 0 and new_score < beta:
                new_score = minimax_with_enhancements(board_copy, depth-1, alpha, beta, 
                                                   True, start_time, time_limit, 
                                                   history_table, killer_moves)[1]
            
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                # Cập nhật killer moves và history table
                killer_moves[depth] = col
                history_table[(col, depth)] = history_table.get((col, depth), 0) + 2**depth
                
                # Lưu cận trên (upper bound)
                transposition_table.store(hash_key, depth, value, column, 2)
                break
                
        # Lưu giá trị chính xác
        transposition_table.store(hash_key, depth, value, column, 0)
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
        
        # Xử lý khi bắt đầu ván mới
        if game_state.is_new_game:
            initialize_tables(len(game_state.board), len(game_state.board[0]))
            print("Bắt đầu ván mới - Đã khởi tạo lại các bảng")
        
        # Update global variables
        global PLAYER_PIECE, AI_PIECE
        AI_PIECE = game_state.current_player
        PLAYER_PIECE = 3 - AI_PIECE
        
        # Verify valid moves
        verified_valid_moves = [col for col in valid_moves if get_next_open_row(board, col) != -1]
        
        if not verified_valid_moves:
            raise ValueError("Không có nước đi hợp lệ sau khi xác minh")
        
        # Use minimax algorithm to select the best move
        selected_col = find_best_move(board, max_depth=6, time_limit=2.0)
        
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
        
        # Kiểm tra xem game đã kết thúc chưa
        if winning_move(board, AI_PIECE):
            print("AI thắng!")
        elif len(get_valid_moves(board)) == 0:
            print("Ván cờ hòa!")
        
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
    initialize_tables()
    uvicorn.run(app, host="0.0.0.0", port=8080)
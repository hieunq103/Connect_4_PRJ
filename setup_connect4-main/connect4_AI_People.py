import numpy as np
import random
import pygame
import sys
import math

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4
MAX_TABLE_SIZE = 1000000

transposition_table = {}

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    if isinstance(board, np.ndarray):
        row_count = len(board)
    else:
        row_count = len(board)
        
    for r in range(row_count-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def drop_piece(board, row, col, piece):
    if row == -1:
        return board
        
    if isinstance(board, np.ndarray):
        board_copy = board.copy()
        board_copy[row][col] = piece
        return board_copy
    else:
        board_copy = [row[:] for row in board]
        board_copy[row][col] = piece
        return board_copy

def winning_move(board, piece):
    # Check horizontal locations for win
    if isinstance(board, np.ndarray):
        row_count = ROW_COUNT
        column_count = COLUMN_COUNT
    else:
        row_count = len(board)
        column_count = len(board[0])
        
    for c in range(column_count-3):
        for r in range(row_count):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(column_count):
        for r in range(row_count-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(column_count-3):
        for r in range(row_count-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(column_count-3):
        for r in range(3, row_count):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    
    return False

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

def get_valid_locations(board):
    if isinstance(board, np.ndarray):
        column_count = COLUMN_COUNT
    else:
        column_count = len(board[0])
    return [col for col in range(column_count) if is_valid_location(board, col)]

def score_position(board, piece):
    if isinstance(board, np.ndarray):
        board_array = board
        row_count = ROW_COUNT
        column_count = COLUMN_COUNT
    else:
        board_array = np.array(board)
        row_count = len(board)
        column_count = len(board[0])
        
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

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
            window = [board_array[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(row_count-3):
        for c in range(column_count-3):
            window = [board_array[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for c in range(column_count):
        for r in range(row_count):
            if board_array[r][c] == piece:
                score += (r + 1) * 0.5
                
    for c in range(column_count):
        r = get_next_open_row(board, c)
        if r == -1:
            continue
            
        if isinstance(board, np.ndarray):
            board_copy = board.copy()
            board_copy[r][c] = piece
        else:
            board_copy = [row[:] for row in board]
            board_copy[r][c] = piece
        
        threat_directions = 0
        
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
        if isinstance(board, np.ndarray):
            board_copy = board.copy()
            board_copy[r][c] = opp_piece
        else:
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
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def manage_transposition_table():
    global transposition_table
    if len(transposition_table) > MAX_TABLE_SIZE:
        transposition_table.clear()

def print_board(board):
    print(board)

def order_moves(board, valid_locations, piece):
    """Order moves by their score in descending order, file 1 version"""
    move_scores = []
    for col in valid_locations:
        row = get_next_open_row(board, col)
        if row != -1:
            temp_board = drop_piece(board, row, col, piece)
            score = score_position(temp_board, piece)
            move_scores.append((col, score))
    move_scores.sort(key=lambda x: x[1], reverse=True)
    ordered_moves = [col for col, score in move_scores]
    return ordered_moves

def sort_valid_moves_with_boards(valid_moves, board, piece):
    """More advanced move sorting with precomputed boards, file 2 version"""
    scored_moves = []
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row != -1:  # Valid move
            board_copy = drop_piece(board, row, col, piece)
            score = score_position(board_copy, piece)
            scored_moves.append((col, score, board_copy))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves

def pick_best_move(board, piece):
    """Simple method to pick the best move based on score (fallback for minimax)"""
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        if row != -1:
            temp_board = drop_piece(board, row, col, piece)
            score = score_position(temp_board, piece)
            if score > best_score:
                best_score = score
                best_col = col

    return best_col

def minimax(board, depth, alpha, beta, maximizing_player):
    valid_moves = get_valid_locations(board)
    
    if isinstance(board, np.ndarray):
        board_tuple = tuple(tuple(row) for row in board)
    else:
        board_tuple = tuple(tuple(row) for row in board)
    
    state_key = (board_tuple, depth, maximizing_player)

    if state_key in transposition_table:
        return transposition_table[state_key]

    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                result = (None, 10000000)
            elif winning_move(board, PLAYER_PIECE):
                result = (None, -1000000)
            else:
                result = (None, 0)
        else:  # Depth is zero
            result = (None, score_position(board, AI_PIECE))
        transposition_table[state_key] = result
        return result
        
    if maximizing_player:
        value = -math.inf
        column = random.choice(valid_moves) if valid_moves else None
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
    else:
        value = math.inf
        column = random.choice(valid_moves) if valid_moves else None
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

def draw_board(board):
    for c in range(COLUMN_COUNT):	
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, 
                               (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), 
                               RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, 
                                   (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), 
                                   RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, 
                                   (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), 
                                   RADIUS)
    pygame.display.update()

board = create_board()
print_board(board)
game_over = False

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER, AI)

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)

        pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
            # Ask for Player 1 Input
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE))

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    # Sửa: Gán lại giá trị board thay vì chỉ gọi hàm
                    board = drop_piece(board, row, col, PLAYER_PIECE)

                    if winning_move(board, PLAYER_PIECE):
                        label = myfont.render("Player 1 wins!!", 1, RED)
                        screen.blit(label, (40,10))
                        game_over = True

                    turn += 1
                    turn = turn % 2

                    print_board(board)
                    draw_board(board)

    if turn == AI and not game_over:				

        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            board = drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (40,10))
                game_over = True

            print_board(board)
            draw_board(board)

            turn += 1
            turn = turn % 2

    if game_over:
        pygame.time.wait(3000)
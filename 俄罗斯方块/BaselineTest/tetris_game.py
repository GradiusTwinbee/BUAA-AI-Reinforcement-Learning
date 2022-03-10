from  malmo import MalmoPython
import os
import sys
import time
from random import randrange as rand
import copy

rewards_map = {'inc_height': -8, 'clear_line': 30, 'holes': -5, 'top_height': -100}

colors = ["Lime_Wool", "Orange_Wool", "Blue_Wool", "Pink_Wool", "Red_Wool",
          "Magenta_Wool", "Yellow_Wool"]

cols = 5
rows = 12

# Define the shapes of the single parts
tetris_shapes = [
	[[0, 1],
     [0, 0]],
	
	[[2, 2],
     [0, 0]],
	
	[[0, 3],
	 [3, 0]],
	
	[[0, 4],
	 [4, 4]],
	
	[[5, 5],
	 [5, 5]]
]

def rotate_clockwise(shape):
    return [ [ shape[y][x]
                    for y in range(len(shape)) ]
            for x in range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

def new_board():
    board = [ [ 0 for x in range(cols) ]
              for y in range(rows) ]
    board += [[ 1 for x in range(cols)]]
    return board

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board
	
def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1	][cx+off_x] += val
    return mat1
    
class TetrisGame:
    def __init__(self, agent_host):
        self.agent_host = agent_host
        self.rlim = cols
        self.next_piece = tetris_shapes[rand(len(tetris_shapes))]
        self.setup()
        self.clear_draw_pieces() #DRAW
        self.start_game()

    def setup(self):
        self.board = new_board()
        self.line_clears = 0
        self.level = 1
        self.new_piece()
        self.lines = 0
        self.score = 0
        self.gameover = False
            
    def new_piece(self):
        self.level += 1
        self.piece = self.next_piece[:]
        self.next_piece = tetris_shapes[rand(len(tetris_shapes))]
        self.piece_x = int(cols/2 - len(self.piece[0])/2)
        self.piece_y = 1

        self.draw_piece2(self.piece, self.piece_x) #DRAW
        if check_collision(self.board, self.piece, (self.piece_x, self.piece_y)):
            self.gameover = True


    def move(self, delta_x):
        ########################################
        ######TODO: fill your code here#########
        ########################################
        if not self.gameover:
            new_x = self.piece_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.piece[0]):
                new_x = cols - len(self.piece[0])
            if not check_collision(self.board,
                                   self.piece,
                                   (new_x, self.piece_y)):
                self.piece_x = new_x

    def drop(self, manual):
        if not self.gameover:
            if check_collision(self.board,
                               self.piece,
                               (self.piece_x, self.piece_y+1)):
                self.piece_y += 1
                self.board = join_matrixes(
                    self.board,
                    self.piece,
                    (self.piece_x, self.piece_y))
                self.new_piece()

                check_board = True
                clear_rows = 0
                while check_board:
                    check_board = False
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            self.draw_piece2(self.board[:-1])
                            self.line_clears += 1
                            check_board = True
                            break
                return True
            else:
                self.erase_piece()
                self.piece_y += 1
                self.draw_piece()
        self.draw_piece2(self.piece, self.piece_x) #DRAW
        return False

    def insta_drop(self):
        if not self.gameover:
            while(not self.drop(True)):
                pass

    def drop_no_draw(self, manual):
        if not self.gameover:
            if check_collision(self.board,
                               self.piece,
                               (self.piece_x, self.piece_y+1)):
                self.piece_y += 1
                self.board = join_matrixes(
                    self.board,
                    self.piece,
                    (self.piece_x, self.piece_y))
                self.new_piece()

                check_board = True
                clear_rows = 0
                while check_board:
                    check_board = False
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            self.line_clears += 1
                            check_board = True
                            break
                return True
            else:
                self.piece_y += 1
        return False

    def insta_drop_no_draw(self):
        if not self.gameover:
            while(not self.drop_no_draw(True)):
                pass
    
    def rotate_piece(self):
        new_piece = rotate_clockwise(self.piece)
        if not check_collision(self.board, new_piece, (self.piece_x, self.piece_y)):
            self.piece = new_piece
            
    def start_game(self):
        if self.gameover:
            self.clear_draw_pieces()  #DRAW
            self.setup()
            self.gameover = False

    def draw_piece(self):
        ########################################
        ######TODO: fill your code here#########
        ########################################
        for cy, row in enumerate(self.piece):
            for cx, col in enumerate(row):
                if col != 0:
                    self.agent_host.sendCommand("chat /setblock " + str(0 + self.piece_x + cx) + " "
                                            + str(80 - self.piece_y - cy) + " 3 wool " + str(col))

    def draw_piece2(self,piece,x=0):
        ########################################
        ######TODO: fill your code here#########
        ########################################

        for cy, row in enumerate(piece):
            for cx, col in enumerate(row):
                if col != 0:
                    self.agent_host.sendCommand("chat /setblock " + str(0 + cx + x) + " "
                                            + str(80 - cy) + " 3 wool " + str(col))
                elif col == 0:
                    self.agent_host.sendCommand("chat /setblock " + str(0 + cx + x) + " "
                                            + str(80 - cy) + " 3 air")

    def erase_piece(self):
        for cy, row in enumerate(self.piece):
            for cx, col in enumerate(row):            
                if col != 0:
                    self.agent_host.sendCommand("chat /setblock " + str(0 + self.piece_x + cx) + " "
                                            + str(80 - self.piece_y - cy) + " 3 air")
    def clear_draw_pieces(self):
        for cy, row in enumerate(self.board[:-1]):
            for cx, col in enumerate(row):
                if (self.board[cy][cx] != 0):
                    self.agent_host.sendCommand("chat /setblock " + str(0 + cx) + " " + str(80 - cy) + " 3 air")

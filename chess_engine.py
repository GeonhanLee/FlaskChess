import chess
import random
import signal
import time
import cProfile
import torch
import torch.nn as nn
import numpy
from create_dataset import chess_board_to_text
from chess_model import LSTM

class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.model = torch.load("model3-1.pt")

    def random_response(self):
        response = random.choice(list(self.board.legal_moves))
        return str(response)

    def select_move(self):
        move, val = self.calculate()
        return str(move)

    def calculate_win_probability(self):
        move, val = self.calculate()
        return val

    def calculate(self):
        legal_moves = list(self.board.legal_moves)

        best_move = legal_moves[0]
        max_val = -999

        for move in legal_moves:
            self.board.push(move)

            # do the calculation
            tensor = chess_board_to_text(str(self.board))
            tensor = tensor.unsqueeze_(0)
            predictions = self.model(tensor.to('cuda'))
            print(predictions)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            np_arr = predictions.detach().cpu().numpy()
            np_arr = np_arr[0].tolist()
            print(np_arr)
            cur_val = np_arr[0]

            if cur_val > max_val:
                max_val = cur_val
                best_move = move
            self.board.pop()
        print(max_val)
        return str(best_move), max_val * 100


# This is being used for testing at the moment, which is why there is so much commented code.
# Will move to a standalone testing script when I get the chance.
if __name__=="__main__":

    fen = "r2qkbr1/ppp1pppp/2n1b2n/8/8/5P2/PPPP2PP/RNB1KBNR b KQq - 0 6"

    newengine = Engine(fen)


    # squares = newengine.board.pieces(1, chess.WHITE)
    # for square in squares:
    #     print (square)
    # print(squares)

    # print(newengine.board)
    # print(newengine.order_moves())

    # print(newengine.material_eval())
    # print(newengine.lazy_eval())

    # start_time = time.time()
    # print(newengine.calculate(3))
    # print(newengine.total_leaves())
    # print("Time taken:", time.time() - start_time)

    start_time = time.time()
    print(newengine.calculate_ab(4))
    print(newengine.total_leaves())
    print("Time taken:", time.time() - start_time)

    start_time = time.time()
    print(newengine.iterative_deepening(4))
    print(newengine.total_leaves())
    print("Time taken:", time.time() - start_time)
    # cProfile.run('newengine.calculate(3)')
    #
    # cProfile.run('newengine.calculate_ab(3)')


    # print(newengine.board)
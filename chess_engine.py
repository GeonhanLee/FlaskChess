import chess
import random
import signal
import time
import cProfile
import torch
import numpy
from create_dataset import chess_board_to_image

class Engine:
    def __init__(self, fen):
        self.board = chess.Board()
        self.board.set_fen(fen)
        self.model = torch.load("model2.pt")

    def random_response(self):
        response = random.choice(list(self.board.legal_moves))
        return str(response)

    def select_move(self):
        legal_moves = list(self.board.legal_moves)

        best_move = legal_moves[0]
        max_index = 0
        max_val = -999

        for move in legal_moves:
            self.board.push(move)

            # do the calculation
            tensor = chess_board_to_image(str(self.board))
            tensor.unsqueeze_(0)
            predictions = self.model(tensor.to('cuda'))

            predictions = torch.nn.functional.softmax(predictions, dim=1)
            np_arr = predictions.detach().cpu().numpy()
            np_arr = np_arr[0].tolist()
            print(np_arr)

            win_prob = np_arr[:5]
            lose_prob = np_arr[5:]

            cur_val = 0
            for idx, var in enumerate(win_prob):
                cur_val += (idx+1) * var
            for idx, var in enumerate(lose_prob):
                cur_val -= (idx+1) * var

            print(cur_val)

            if cur_val > max_val:
                max_val = cur_val
                best_move = move

            self.board.pop()

        return str(best_move)



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
import chess
import csv
import chess.pgn
import torch


def chess_board_to_image(board_str):
    image = [[] for i in range(8)]

    for i, row in enumerate(board_str.split('\n')):
        for char in row:
            if char != ' ':
                piece_index = {
                    'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
                    'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12
                }.get(char, 0)  # 0 for empty
                image[i].append(piece_index)
    tensor = torch.tensor(image, dtype=torch.float)
    tensor *= 10
    tensor.unsqueeze_(0)
    return tensor

def train_with_pgn(dataset, labelset):
    pgn = open("Nakamura.pgn")
    cur_game = chess.pgn.read_game(pgn)

    while cur_game is not None:
        if len(dataset) > 100000:
            return
        else:
            if len(dataset) % 10 == 0:
                print(len(dataset))

        result = cur_game.headers['Result']

        if result != '1-0' and result != '0-1':
            cur_game = chess.pgn.read_game(pgn)
            continue

        move_count = 0
        while cur_game is not None:
            board = cur_game.board()

            if board.turn == chess.WHITE:
                tensor = chess_board_to_image(str(board))
                dataset.append(tensor)
                move_count += 1

            cur_game = cur_game.next()

        if result == '0-1':
            for i in range(move_count):
                labelset.append(int(i / move_count * 5))
        if result == '1-0':
            for i in range(move_count):
                labelset.append(5 + int(i / move_count * 5))
        cur_game = chess.pgn.read_game(pgn)

if __name__ == "__main__":
    images = []
    labels = []

    train_with_pgn(images, labels)

    print(len(images))
    print(len(labels))

    torch.save(images, 'boards2.pt')
    torch.save(labels, 'labels2.pt')



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


def chess_board_to_text(board_str):
    text = []

    for i, row in enumerate(board_str.split('\n')):
        for char in row:
            if char != ' ':
                piece_index = {
                    'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
                    'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12
                }.get(char, 0)  # 0 for empty
                text.append(piece_index)
    tensor = torch.tensor(text, dtype=torch.int)
    return tensor


def train_with_pgn(path, dataset, labelset):
    pgn = open(path)
    cur_game = chess.pgn.read_game(pgn)
    game_count = 0

    while cur_game is not None:
        if game_count > 20000:
            break
        else:
            if game_count % 100 == 0:
                print(game_count)

        result = cur_game.headers['Result']

        if result != '1-0' and result != '0-1':
            cur_game = chess.pgn.read_game(pgn)
            continue
        game_count += 1
        move_count = 0
        while cur_game is not None:
            board = cur_game.board()

            if board.turn == chess.WHITE:
                tensor = chess_board_to_text(str(board))
                dataset.append(tensor)
                move_count += 1

            cur_game = cur_game.next()

        if result == '0-1':
            for i in range(move_count):
                labelset.append(0.5 + (i / move_count) * 0.5)
        if result == '1-0':
            for i in range(move_count):
                labelset.append(0.5 - (i / move_count) * 0.5)
        cur_game = chess.pgn.read_game(pgn)

    print("game count of ", game_count)

if __name__ == "__main__":
    images = []
    labels = []

    train_with_pgn("lichess_db_standard_rated_2013-01.pgn", images, labels)

    print(len(images))
    print(len(labels))

    torch.save(images, 'boards3-1.pt')
    torch.save(labels, 'labels3-1.pt')



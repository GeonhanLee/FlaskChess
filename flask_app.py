from flask import Flask, render_template
from chess_engine import *
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/move/<int:depth>/<path:fen>/')
def get_move(depth, fen):
    print(fen)
    print(depth)
    print("Calculating...")
    engine = Engine(fen)
    move = engine.select_move()
    return move


@app.route('/test/<string:tester>')
def test_get(tester):
    return tester


if __name__ == '__main__':
    app.run(debug=True)
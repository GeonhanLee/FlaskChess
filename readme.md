# Lstm binary classification 모델을 활용한 체스 AI

## 데이터셋 설명
체스 게임의 데이터를 담은 pgn 파일에서 기물 포지션을 토큰 형태로 변환합니다. 
2만개의 게임에서 약 60만개의 포지션을 학습했습니다.

처음에는 단순히 화이트가 이긴 포지션은 0, 블랙이 이긴 포지션은 1로 레이블링을 하였지만 결과가 잘 나오지 않았습니다.

이를 해결 해보기 위해 게임 초반의 움직임보다 후반의 움직임이 승률에 영향이 더 클것이라 가정하고, 
게임의 초반 승률은 0.5, 후반으로 갈수록 승률이 1에 가까워지도록 레이블링을 했습니다. 

이전보다는 조금더 체스를 잘하지만 그래도 아쉬운 부분이 존재합니다. 
수적 우위를 점하기 위해 플레이어의 기물을 잡는 판단은 잘하지만 자신의 기물이 잡힌다는 개념을 잘 모르는 듯 합니다. 

## 수 놓는 기준
기물을 이동했을때의 포지션의 승률을 확인합니다. 
모든 경우의 수를 확인후 가장 승률이 높은 무브를 선택합니다. 
ai가 판단한 승률을 확인 가능합니다. 

## 프론트엔드, 체스 엔진
FlaskChess 레포지토리를 포크하여 수정하였습니다. 
Flask 백엔드에서 AI 모델을 로드하고 실행합니다. 

## 참고한 코드
https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb 

https://github.com/LukeDitria/pytorch_tutorials/blob/main/section12_sequential/solutions/Pytorch6_LSTM_Text_Classification.ipynb 

데이터 출처 
https://database.lichess.org/ 

# original readme:

This is a simple chess engine/interface created using flask. 
It uses chessboard.js and chess.js for the logic of the frontend chessboard, and python chess for the 
logic of the backend chessboard. All calculation is done on the backend using python. 

In order to run this application on your own machine, please install flask and python chess. 

Install flask by running: 
    pip install flask 

Install python chess by running: 
    pip install python-chess[uci,gaviota] 

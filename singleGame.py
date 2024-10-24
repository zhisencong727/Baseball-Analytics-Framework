from pybaseball import statcast_single_game
import os

game_id = 663178
data = statcast_single_game(game_id)


fileName = str(game_id) + ".csv"
fileName = os.path.join("singleGameFolder",fileName)
data.to_csv(fileName,index=True)
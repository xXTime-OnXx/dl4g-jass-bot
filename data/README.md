# Data Sets

**Data originally from Swisslos Jass Platform is available:**
- under Data/games on Illias,
- and on the /exchange/dl4g/data/games folder on the GPU cluster
- Each data file contains 100000 games
- Each line is a complete game in json format that can be read by

**GameLogEntry.from_json() to get an entry:**
- An entry consists of:
- The game (as GameState for a complete game)
- The date
- The player ids on the server, id=0 for anonymous play
- Data may not be used outside of HSLU


## Games

The data for the games are available on ilias due to large size

https://elearning.hslu.ch/ilias/ilias.php?baseClass=ilrepositorygui&cmd=view&ref_id=6377653


## Statistics

- Statistics about the player ids is available in the file stat/player_all_stat.json
- It could be used to filter games for training, for example only using ids of the better than average players

## Card Prediction

The directory `card_prediction` stores the dataset for the card play training.

Datasets:
| Version    | File | Description |
| -------- | ------- | ------------------------------------ |
| v1  | `card_prediction_0001.csv`    | This is the first dataset version to train a simple card_play model |
| v2 | `card_prediction_0002.csv`    | This dataset is based on v1 but uses the column order that is given by the `jasskit-library` to simplify the usage of the model. |

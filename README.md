# Movie Classifier

# Installation
Install the dependencies using pip:
`pip install . --upgrade`

# Training
1. Download the dataset from [here](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) and put the csv file in the dataset folder.
2. Download the pre-trained word embeddings from [here]() and put the txt file in the dataset folder.
3. Run: `python model.py --mode train --model_path model --embeddings_size 100 --max_length 128`

# Inference
`python model.py --mode classify --model_path model --title 'Inception' --description "Dom Cobb is a thief with the rare ability to enter people's dreams and steal their secrets from their subconscious. His skill has made him a hot commodity in the world of corporate espionage but has also cost him everything he loves. Cobb gets a chance at redemption when he is offered a seemingly impossible task: Plant an idea in someone's mind. If he succeeds, it will be the perfect crime, but a dangerous enemy anticipates Cobb's every move."`

Output:
`[('Action', 0.7449887), ('Thriller', 0.6140004), ('Crime', 0.4795791), ('Comedy', 0.4772842)]`

# Evaluation

|label|precision|recall|F1|support|
|---|---|---|---|---|
| Action          | 0.577     | 0.420  | 0.486 | 1049    |
| Adventure       | 0.440     | 0.213  | 0.287 | 522     |
| Animation       | 0.652     | 0.274  | 0.386 | 548     |
| Comedy          | 0.595     | 0.633  | 0.613 | 2552    |
| Crime           | 0.424     | 0.253  | 0.317 | 596     |
| Documentary     | 0.879     | 0.534  | 0.664 | 1101    |
| Drama           | 0.627     | 0.712  | 0.667 | 3482    |
| Family          | 0.522     | 0.287  | 0.370 | 541     |
| Fantasy         | 0.496     | 0.128  | 0.203 | 493     |
| Foreign         | 0.091     | 0.003  | 0.006 | 333     |
| History         | 0.370     | 0.135  | 0.197 | 223     |
| Horror          | 0.727     | 0.532  | 0.614 | 882     |
| Music           | 0.606     | 0.355  | 0.448 | 290     |
| Mystery         | 0.396     | 0.144  | 0.211 | 382     |
| Romance         | 0.514     | 0.419  | 0.462 | 1037    |
| Science Fiction | 0.680     | 0.475  | 0.560 | 612     |
| TV Movie        | 0.500     | 0.004  | 0.008 | 249     |
| Thriller        | 0.509     | 0.312  | 0.387 | 1179    |
| War             | 0.560     | 0.365  | 0.442 | 178     |
| Western         | 0.692     | 0.512  | 0.589 | 123     |

|metric|value|
|---|---|
| Micro Average Precision | 0.605 |
| Micro Average Recall | 0.466 |
| Micro Average F1 | 0.526 |
| Accuracy | 0.226 |
| Macro Average Precision | 0.585 |
| Macro Average Precision (not weighted) | 0.542 |
| Macro Average Recall | 0.466 |
| Macro Average Recall (not weighted) | 0.335 |
| Macro Average F1 | 0.501 |
| Macro Average F1 (not weighted) | 0.395 |

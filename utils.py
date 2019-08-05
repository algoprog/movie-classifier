import csv
import json
import numpy


def load_dataset(path, min_frequency=300):
    """
    Loads the dataset
    :param min_frequency: the minimum frequency of every label
    :param path: path to the dataset
    :return: lists of titles, descriptions and types
    """
    titles = []
    descriptions = []
    types = []
    types_freq = {}

    file = open(path)
    data = csv.DictReader(file)
    for row in data:
        genres = json.loads(row['genres'].replace("'", '"'))
        for genre in genres:
            genre = genre['name']
            if genre in types_freq:
                types_freq[genre] += 1
            else:
                types_freq[genre] = 1

    file = open(path)
    data = csv.DictReader(file)
    for row in data:
        title = row['original_title'].lower()
        description = row['overview'].lower()

        genre = json.loads(row['genres'].replace("'", '"'))
        genre = [g['name'] for g in genre if types_freq[g['name']] > min_frequency]

        if len(genre) > 0:
            titles.append(title)
            descriptions.append(description)
            types.append(genre)

    return titles, descriptions, types


class JsonEncoder(json.JSONEncoder):
    """ A custom JSON encoder that encodes float and int values properly """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

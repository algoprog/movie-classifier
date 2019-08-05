import argparse
import operator
import pickle
import random
import keras

import numpy as np

from flask import Flask, request, json
from flask_cors import CORS
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Concatenate
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from texttable import Texttable
from datetime import datetime
from utils import load_dataset, JsonEncoder

random.seed(42)


def load_pretrained_word_embeddings(path, tokenizer, embeddings_size):
    """
    Creates an embedding matrix from pre-trained word embeddings
    :param path: the path to the word embeddings txt file
    :param tokenizer: a Tokenizer object
    :param embeddings_size: the size of the embeddings
    :return:
    """
    print('Loading word vectors')

    f = open(path, 'r', encoding='utf8')
    word_vectors = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        word_vectors[word] = embedding

    print(len(word_vectors), ' words loaded!')

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embeddings_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = word_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_model(tokenizer, total_labels, embeddings_size, max_length=128, inference=False):
    """
    Builds the Keras multi-class multi-label classification model
    :param tokenizer: a Tokenizer object
    :param total_labels: the total number of unique labels
    :param embeddings_size: the size of the embeddings
    :param max_length: the maximum sequence length
    :param inference: False for training, True for predictions
    :return: Keras model
    """
    vocab_size = len(tokenizer.word_index) + 1

    input_title = Input(shape=(max_length,))
    input_description = Input(shape=(max_length,))

    if not inference:
        embedding_matrix = load_pretrained_word_embeddings('dataset/glove.6B.100d.txt', tokenizer, embeddings_size)
        word_embeddings = Embedding(input_dim=vocab_size,
                                    output_dim=embeddings_size,
                                    weights=[embedding_matrix],
                                    mask_zero=True,
                                    input_length=max_length,
                                    trainable=False)
    else:
        word_embeddings = Embedding(input_dim=vocab_size,
                                    output_dim=embeddings_size,
                                    mask_zero=True,
                                    input_length=max_length,
                                    trainable=False)

    title_embedding = Bidirectional(LSTM(units=64,
                                         activation='tanh',
                                         return_sequences=False))(word_embeddings(input_title))
    description_embedding = Bidirectional(LSTM(units=64,
                                               activation='tanh',
                                               return_sequences=False))(word_embeddings(input_description))

    movie_embedding = Concatenate()([title_embedding, description_embedding])

    output = Dense(total_labels, activation='sigmoid')(movie_embedding)

    model = Model(inputs=[input_title, input_description], outputs=[output])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # model.summary()

    return model


def evaluate(model, val_data, label_binarizer):
    """
    Calculates evaluation metrics for the model:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1: 2 * precision * recall / (precision + recall)
        micro metric: calculated per sample
        macro metric: calculated by averaging the per class metrics
        accuracy: % of samples with all the classes correct
    :param model: Keras model
    :param val_data: dict with validation data
    :param label_binarizer: a MultiLabelBinarizer object
    :return: precision, recall, f1, acc, macro avg precision, macro avg recall, macro avg f1
    """
    print('\nevaluation...\n')
    val_titles, val_descriptions, val_types = val_data['titles'], val_data['descriptions'], val_data['types']

    y_true = val_types
    y_pred = np.round(model.predict([val_titles, val_descriptions]))

    total_samples = len(y_true)
    total_labels = len(label_binarizer.classes_)

    types_occurrences = {}
    for i in range(total_samples):
        for j in range(total_labels):
            if val_types[i][j] == 1:
                if j in types_occurrences:
                    types_occurrences[j] += 1
                else:
                    types_occurrences[j] = 1

    correct = 0
    true_positives = {}
    predicted_positives = {}
    possible_positives = {}
    for i in range(total_labels):
        true_positives[i] = 0
        predicted_positives[i] = 0
        possible_positives[i] = 0

    for i in range(total_samples):
        if np.array_equal(y_true[i], y_pred[i]):
            correct += 1
        for j in range(total_labels):
            if y_true[i][j] == 1:
                possible_positives[j] += 1
                if y_pred[i][j] == 1:
                    true_positives[j] += 1
            if y_pred[i][j] == 1:
                predicted_positives[j] += 1

    acc = correct / total_samples

    avg_precision = 0
    avg_precision_nw = 0
    avg_recall = 0
    avg_recall_nw = 0
    avg_f1 = 0
    avg_f1_nw = 0
    weights_sum = 0

    t = Texttable()

    t.add_row(['label', 'precision', 'recall', 'F1', 'support'])

    for i in range(total_labels):
        if predicted_positives[i] == 0:
            precision = 0
        else:
            precision = true_positives[i] / predicted_positives[i]

        if possible_positives[i] == 0:
            recall = 0
        else:
            recall = true_positives[i] / possible_positives[i]

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        support = types_occurrences[i]

        t.add_row([label_binarizer.classes_[i], precision, recall, f1, support])

        weight = support / total_samples
        weights_sum += weight

        avg_precision += weight * precision
        avg_precision_nw += (1 / total_labels) * precision
        avg_recall += weight * recall
        avg_recall_nw += (1 / total_labels) * recall
        avg_f1 += weight * f1
        avg_f1_nw += (1 / total_labels) * f1

    avg_precision /= weights_sum
    avg_recall /= weights_sum
    avg_f1 /= weights_sum
    precision = sum(true_positives.values()) / sum(predicted_positives.values())
    recall = sum(true_positives.values()) / sum(possible_positives.values())
    f1 = 2 * (precision * recall) / (precision + recall)

    print(t.draw())

    print('---')
    print('Micro Average Precision:', precision)
    print('Micro Average Recall:', recall)
    print('Micro Average F1:', f1)
    print('Accuracy:', acc)
    print('Macro Average Precision:', avg_precision)
    print('Macro Average Precision (not weighted):', avg_precision_nw)
    print('Macro Average Recall:', avg_recall)
    print('Macro Average Recall (not weighted)', avg_recall_nw)
    print('Macro Average F1:', avg_f1)
    print('Macro Average F1 (not weighted):', avg_f1_nw)
    print('---')

    return precision, recall, f1, acc, avg_precision, avg_recall, avg_f1


def train(max_length=128, embeddings_size=100, validation_split=0.2, model_path='model'):
    """
    Trains the model
    :param model_path: the path to the folder containing the model
    :param validation_split: percentage of the samples kept for validation
    :param max_length: maximum sequence length
    :param embeddings_size: the size of the embeddings
    :return:
    """
    titles, descriptions, types = load_dataset('dataset/movies_metadata.csv')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles + descriptions)

    label_binarizer = MultiLabelBinarizer()
    types = label_binarizer.fit_transform(types)

    with open(model_path + '/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(model_path + '/label_binarizer.pkl', 'wb') as f:
        pickle.dump(label_binarizer, f)

    # Converts texts to sequences of token ids
    titles = tokenizer.texts_to_sequences(titles)
    descriptions = tokenizer.texts_to_sequences(descriptions)

    # Pads the sequences with zeros
    titles = pad_sequences(titles, padding='post', maxlen=max_length)
    descriptions = pad_sequences(descriptions, padding='post', maxlen=max_length)

    # Split the dataset to train and validation
    train_num = int((1 - validation_split) * len(titles))
    train_titles, train_descriptions, train_types = titles[:train_num], descriptions[:train_num], types[:train_num]
    val_titles, val_descriptions, val_types = titles[train_num:], descriptions[train_num:], types[train_num:]

    total_labels = len(label_binarizer.classes_)

    model = get_model(embeddings_size=embeddings_size, tokenizer=tokenizer, total_labels=total_labels)

    class MyCallback(keras.callbacks.Callback):
        """ A custom Keras callback for running the evaluation every k batches and storing the best model """
        def __init__(self, model, val_data, label_binarizer):
            super(MyCallback, self).__init__()
            self.model = model
            self.val_data = val_data
            self.label_binarizer = label_binarizer
            self.f1 = 0

        def on_batch_end(self, batch, logs={}):
            if (batch + 1) % 100 == 0:
                precision, recall, f1, acc, avg_precision, avg_recall, avg_f1 = evaluate(
                    self.model,
                    val_data=self.val_data,
                    label_binarizer=self.label_binarizer
                )
                logs['micro precision'] = precision
                logs['micro recall'] = recall
                logs['micro F1'] = f1
                logs['macro precision'] = avg_precision
                logs['macro recall'] = avg_recall
                logs['macro F1'] = avg_f1
                logs['accuracy'] = acc
                if f1 > self.f1:
                    print(str(self.f1) + ' -> ' + str(f1))
                    self.f1 = f1
                    model.save('model/model.h5')

    # A tensorboard callback to visualize the metrics
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        update_freq='batch',
        write_grads=True,
        write_graph=True,
        write_images=True
    )
    callback = MyCallback(model=model,
                          val_data={'titles': val_titles,
                                    'descriptions': val_descriptions,
                                    'types': val_types},
                          label_binarizer=label_binarizer)

    model.fit([train_titles, train_descriptions], [train_types],
              epochs=32,
              verbose=True,
              batch_size=64,
              callbacks=[callback, tensorboard_callback])


def load_model(model_path='model'):
    """
    Loads the Keras model, the weights, the tokenizer and the label binarizer
    :param model_path: the path to the folder containing the model
    :return: model, tokenizer, label_binarizer
    """
    f = open(model_path + '/tokenizer.pkl', 'rb')
    tokenizer = pickle.load(f)

    f = open(model_path + '/label_binarizer.pkl', 'rb')
    label_binarizer = pickle.load(f)
    total_labels = len(label_binarizer.classes_)

    model = get_model(embeddings_size=100, tokenizer=tokenizer, total_labels=total_labels, inference=True)
    model.load_weights(model_path + '/model.h5')
    model._make_predict_function()

    return model, tokenizer, label_binarizer


def inference(model, tokenizer, label_binarizer, input, max_length=128, min_probability=0.4):
    """
    Runs the model for a given input and returns a list of labels with their probability
    :param model: the Keras model
    :param tokenizer: a Tokenizer object
    :param label_binarizer: a MultiLabelBinarizer object
    :param input: a dict {'title':<str>, 'description':<str>}
    :param max_length: the maximum sequence length
    :param min_probability: the minimum required probability to display a label
    :return:
    """
    title = tokenizer.texts_to_sequences([input['title'].lower()])
    description = tokenizer.texts_to_sequences([input['description'].lower()])

    title = pad_sequences(title, padding='post', maxlen=max_length)
    description = pad_sequences(description, padding='post', maxlen=max_length)

    y = model.predict([title, description])
    probabilities = y[0]
    result = {}
    for i, probability in enumerate(probabilities):
        if probability > min_probability:
            result[label_binarizer.classes_[i]] = probability

    result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-mode", "--mode", required=True,
                    help="'train' for training, 'classify' for classification, 'serve' for web API")
    ap.add_argument("-model_path", "--model_path", required=False,
                    help="the folder where the model files will be stored", default='model')
    ap.add_argument("-embeddings_size", "--embeddings_size", required=False,
                    help="the size of the embedding layer", default=100)
    ap.add_argument("-max_length", "--max_length", required=False,
                    help="max length of the title and description", default=128)
    ap.add_argument("-title", "--title", required=False,
                    help="the title of the movie to be classified", default='')
    ap.add_argument("-description", "--description", required=False,
                    help="the description of the movie to be classified", default='')
    ap.add_argument("-port", "--port", required=False,
                    help="port to serve the model", default=80)
    args = vars(ap.parse_args())

    if args['mode'] == 'train':
        train(max_length=args['max_length'],
              embeddings_size=args['embeddings_size'],
              validation_split=0.2,
              model_path=args['model_path'])

    elif args['mode'] == 'classify':
        if args['title'] == '':
            print('Please enter the title of the movie')
            exit()
        if args['description'] == '':
            print('Please enter the description of the movie')
            exit()

        model, tokenizer, label_binarizer = load_model(args['model_path'])

        result = inference(
            model=model,
            tokenizer=tokenizer,
            label_binarizer=label_binarizer,
            input={
                'title': args['title'],
                'description': args['description']
            }
        )

        print(result)

    elif args['mode'] == 'serve':
        app = Flask(__name__)
        CORS(app)

        model, tokenizer, label_binarizer = load_model(args['model_path'])

        @app.route('/classify', methods=['POST', 'GET'])
        def classify():
            title = request.args.get('title')
            description = request.args.get('description')
            if title is None or description is None:
                return ''

            result = inference(
                model=model,
                tokenizer=tokenizer,
                label_binarizer=label_binarizer,
                input={
                    'title': title,
                    'description': description
                }
            )

            return json.dumps({'genres': result}, cls=JsonEncoder)

        app.run(host='0.0.0.0', port=args['port'])

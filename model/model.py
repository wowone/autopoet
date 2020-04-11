import logging
import re
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class Model:
    def __init__(self, seq_size, batch_size, tokenizer):
        self.sequence_size = seq_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model = None
        self.x_test = None
        self.y_test = None

    def shuffle_and_split_training_set(self, sentences_original, next_original, test_size=0.02):
        # shuffle at unison
        # print('Shuffling sentences')

        tmp_sentences = []
        tmp_next_word = []
        for i in np.random.permutation(len(sentences_original)):
            tmp_sentences.append(sentences_original[i])
            tmp_next_word.append(next_original[i])

        cut_index = int(len(sentences_original) * (1. - test_size))
        x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
        y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]
        self.x_test, self.y_test = x_test, y_test

        # print("Size of training set = %d" % len(x_train))
        # print("Size of test set = %d" % len(y_test))
        return x_train, y_train, x_test, y_test

    # TODO: rewrite this
    @staticmethod
    def generate_probas(preds, temperature):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        # probas = np.random.multinomial(1, preds, 1)  # Why multinomial?
        return preds

    def get_word(self, prediction, temperature=1.0):
        p = self.generate_probas(prediction, temperature)
        return self.tokenizer.id_to_word[np.argmax(p)]

    # Data generator for fit and evaluate
    def generator(self, sentence_list, next_word_list, batch_size):
        index = 0
        while True:
            x = np.zeros((batch_size, self.sequence_size))
            y = np.zeros(batch_size)
            for i in range(batch_size):
                x[i] = sentence_list[index % len(sentence_list)]
                # for t, word_id in enumerate(sentence_list[index % len(sentence_list)]):
                #    x[i, t, word_id] = 1
                y[i] = next_word_list[index % len(sentence_list)]
                index += 1
            yield x, to_categorical(y, num_classes=self.tokenizer.vocab_size)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.tokenizer.vocab_size, 50, input_length=self.sequence_size))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.tokenizer.vocab_size, activation='softmax'))
        print(model.summary())
        self.model = model

    def on_epoch_end(self, epoch, logs):
        # select a seed text
        seed_text = self.x_test[np.random.randint(len(self.x_test))]
        seed_text_reversed = self.tokenizer.ids_to_text(reversed(seed_text))
        seed_text = self.tokenizer.ids_to_text(seed_text)

        # generate new text
        generated = self.generate_seq(seed_text, 10)
        generated_reversed = ' '.join(reversed(generated))
        print(f"Generated: {generated_reversed}\nOn seed: {seed_text_reversed}\n")

    def generate_seq(self, seed_text, n_words):
        result = list()
        in_text = seed_text
        # generate a fixed number of words

        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.text_to_seq([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=self.sequence_size, truncating='pre')
            # predict probabilities for each word
            prediction = self.model.predict(encoded)[0]
            # get word by prediction (just argmax)
            # TODO: use beam search here?
            predicted_word = self.get_word(prediction)
            result.append(predicted_word)
            in_text += ' ' + predicted_word
        return result

    def compile_model(self):
        # compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit_model(self, data_x, data_y, test_x, test_y, epochs=150):
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        self.model.fit_generator(self.generator(data_x, data_y, self.batch_size),
                                 steps_per_epoch=int(len(data_x) / self.batch_size) + 1,
                                 epochs=epochs,
                                 callbacks=[print_callback],
                                 validation_data=self.generator(test_x, test_y, self.batch_size),
                                 validation_steps=int(len(test_x) / self.batch_size) + 1)

    def save_model(self, filename='model.h5'):
        self.model.save(filename)

    def load_model(self, model):
        self.model = model

    def split_to_sequences(self, text_as_list):
        sequences, next_words = [], []
        for i in range(len(text_as_list) - self.sequence_size):
            sequence = text_as_list[i + 1:i + 1 + self.sequence_size][::-1]
            next_words.append(text_as_list[i])
            sequences.append(sequence)
        logging.info('Total Sequences: %d' % len(sequences))
        return sequences, next_words


# TODO: Move function to lingtools
def clean_text(text):
    text = text.lower()
    return re.sub(r'[^А-Яа-я\s\n]', '', text)





# if __name__ == '__main__':
#     # Local env:
#     path_to_text = "../pushkin.txt"
#
#     # model configs:
#     seq_size = 32
#     batch_size = 50
#
#
#     # Load text_data:
#     with open(path_to_text, encoding='utf-8') as f:
#         text = f.read()
#     text = clean_text(text)
#     tokens = word_tokenize(text)
#     words = set(tokens)
#
#     # Split text to seq:
#     (lines, next_words) = build_sequences(tokens, words, seq_size)
#
#     # Map text to tokens
#     tokenizer = tokenizer.MyTokenizer(clean_text, word_tokenize)
#     tokenizer.fit(text)
#     sequences = tokenizer.text_to_seq(lines)  # That's important, that it's lines, not text
#     next_words = list(map(lambda x: tokenizer.word_to_id[x], next_words))
#
#     # Create model
#     my_model = Model(seq_size, batch_size, tokenizer)
#
#     # separate into input and output
#     sequences = np.array(sequences)
#     next_words = np.array(next_words)
#     seq_length = seq_size
#     (X, y), (X_test, y_test) = my_model.shuffle_and_split_training_set(
#         sequences[:, :-1], next_words
#     )
#
#     my_model.build_model()
#     my_model.compile_model()
#     my_model.fit_model(X, y, X_test, y_test)
#     my_model.save_model()
#
#     # save the tokenizer
#     dump(tokenizer, open('tokenizer.pkl', 'wb'))

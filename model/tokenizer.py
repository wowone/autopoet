from nltk.tokenize import word_tokenize


class MyTokenizer:
    def __init__(self, clean_text):
        self.id_to_word = {}
        self.word_to_id = {}
        self.cleaner = clean_text
        self.tokenizer = word_tokenize
        self.vocab_size = None

    def tokenize(self, text):
        return self.tokenizer(text)

    def fit(self, text_to_fit):
        tokenized_text = self.tokenize(text_to_fit)
        tokens = sorted(set(tokenized_text))
        self.id_to_word = {i: v for i, v in enumerate(tokens)}
        self.word_to_id = {v: i for i, v in enumerate(tokens)}
        self.vocab_size = len(self.word_to_id)

    def text_to_ids(self, text):
        if isinstance(text, list):
            token_list = text
        elif isinstance(text, str):
            token_list = self.tokenize(text)
        else:
            raise TypeError("Expected either list or str argument")
        return [self.word_to_id[token] for token in token_list]

    def ids_to_text(self, ids):
        return ' '.join([self.id_to_word[id] for id in ids])

    def text_to_seq(self, lines):
        res = []
        for line in lines:
            line_res = []
            tokens = self.tokenizer(line)
            for word in tokens:
                line_res.append(self.word_to_id[word])
            res.append(line_res)
        return res

import collections

class MyTokenizer:
    def __init__(self, clean_text, word_tokenizer):
        self.id_to_word = {}
        self.word_to_id = {}
        self.cleaner = clean_text
        self.tokenizer = word_tokenizer
        self.vocab_size = None

    def set_cleaner(self, cleaner):
        self.cleabner = cleaner

    def set_tokenizer(self, token_f):
        self.tokenizer = token_f

    def fit(self, text_to_fit):
        text_to_fit = self.cleaner(text_to_fit)
        tokens = set(self.tokenizer(text_to_fit))
        tokens = sorted(tokens)
        self.id_to_word = {i: v for i, v in enumerate(tokens)}
        self.word_to_id = {v: i for i, v in enumerate(tokens)}
        self.vocab_size = len(self.word_to_id)

    def text_to_seq(self, lines):
        res = []
        for line in lines:
            line_res = []
            tokens = self.tokenizer(line)
            for word in tokens:
                line_res.append(self.word_to_id[word])
            res.append(line_res)
        return res
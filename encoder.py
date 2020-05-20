import re
import torch
import numpy as np
from bigfile import BigFile


class Text2BoWEncoder:
    def __init__(self, vocab_file):
        self.word2index = {
            line.split(" ", 1)[0]: i
            for i, line in enumerate(open(vocab_file).readlines())
        }
        self.n_dims = len(self.word2index)

    def encode(self, string):
        words = self.tokenize(string)
        vector = np.zeros([self.n_dims])
        for word in words:
            idx = self.word2index.get(word, None)
            if idx is not None:
                vector[idx] += 1
        return torch.Tensor(vector)

    @staticmethod
    def tokenize(string):
        string = string.replace('\r', ' ')
        string = re.sub(r"[^A-Za-z0-9]", " ", string).strip().lower()
        words = string.split()
        return words


class Text2W2VEncoder:
    def __init__(self, data_path):
        self.w2v = BigFile(data_path)
        vocab_size, self.ndims = self.w2v.shape()
        print("Text2W2VEncoder", "vocab_size", vocab_size, "dim", self.ndims)

    def encode(self, words):
        renamed, vectors = self.w2v.read(words)

        if len(vectors) > 0:
            vec = np.array(vectors).mean(axis=0)
        else:
            vec = np.zeros([self.ndims])
        return torch.Tensor(vec)

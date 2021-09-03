# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.data import Dataset, DataLoader


def get_vocab(vocab_path="./data/couplet/vocabs"):
    vocabs = []
    with open(vocab_path) as f:
        for line in f:
            vocabs.append(line.strip())
    word2id = {word: i for i, word in enumerate(vocabs)}
    id2word = {i: word for i, word in enumerate(vocabs)}
    return word2id, id2word, len(vocabs)


class CoupletDataSet(Dataset):
    def __init__(self, data_dir, max_seq_len=30):
        self.in_path = os.path.join(data_dir, "in.txt")
        self.out_path = os.path.join(data_dir, "out.txt")
        self.word2id, self.id2word, vocab_size = get_vocab()
        self.max_seq_len = max_seq_len
        self.X = self.pad_sentences(self.in_path)
        self.Y = self.pad_sentences(self.out_path, is_in=False)

    def pad_sentences(self, file_path, is_in=True):
        ids = []
        with open(file_path) as f:
            for line in f:
                lis = line.strip().split()
                if not is_in:
                    lis = ["<bos>"] + lis + ["<eos>"]
                tmp_ids = [self.word2id.get(w, self.word2id["<unk>"]) for w in lis[:self.max_seq_len]]
                if len(tmp_ids) < self.max_seq_len:
                    tmp_ids += [self.word2id.get("<pad>")] * (self.max_seq_len-len(tmp_ids))
                ids.append(tmp_ids)
        return torch.LongTensor(ids)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    dataset = CoupletDataSet("./data/couplet/test")
    dataloader = DataLoader(dataset, batch_size=2)
    for a, b in dataloader:
        print(a)
        print(b)
        break

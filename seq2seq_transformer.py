# -*- coding: utf-8 -*-

import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_helper import CoupletDataSet, get_vocab


WORD2ID, ID2WORD, VOCAB_SIZE = get_vocab()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD_IDX = WORD2ID["<pad>"]
UNK_IDX = WORD2ID["<unk>"]
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
LR = 0.0001
EPOCHS = 10
MODEL_SAVE_PATH = "./model/model_transformer.pth"


class PositionalEncoding(nn.Module):
    """transformer 位置编码"""
    def __init__(self, embed_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embed_size, 2) * math.log(10000) / embed_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, token_embedding.size(1), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.embed_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embed_size,
                 nhead, vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.generator = nn.Linear(embed_size, vocab_size)
        self.tok_emb = TokenEmbedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.pos_encoding(self.tok_emb(src))
        tgt_emb = self.pos_encoding(self.tok_emb(tgt))

        outputs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                   src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outputs)

    def encoder(self, src, src_mask):
        return self.transformer.encoder(self.pos_encoding(self.tok_emb(src)), src_mask)

    def decoder(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.pos_encoding(self.tok_emb(tgt)), memory, tgt_mask)


# 生成mask
def generate_square_subsequent_mask(sz):
    """
    :param sz:
    :return: 0, False表示能看到
    tensor([[0., inf, inf, inf, inf],
        [0., 0., inf, inf, inf],
        [0., 0., 0., inf, inf],
        [0., 0., 0., 0., inf],
        [0., 0., 0., 0., 0.]])
    """
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # 解码过程中，decoder attention 不能看到后面文字
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # encoder attention 可以看到所有文字
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train():
    transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                     num_decoder_layers=NUM_DECODER_LAYERS,
                                     embed_size=EMB_SIZE,
                                     nhead=NHEAD,
                                     vocab_size=VOCAB_SIZE,
                                     dim_feedforward=FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if os.path.exists(MODEL_SAVE_PATH):
        transformer.load_state_dict(torch.load(MODEL_SAVE_PATH))

    transformer = transformer.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    train_dataset = CoupletDataSet("./data/couplet/train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CoupletDataSet("./data/couplet/test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_test_loss = float('inf')

    for epoch in range(1, EPOCHS+1):
        transformer.train()
        train_loss = 0.
        for i, (src, tgt) in enumerate(train_dataloader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                                 tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 0:
                print("TRAIN: EPOCH: {} STEP: {} LOSS: {}".format(epoch, i, loss.item()))

        train_loss = train_loss / len(train_dataloader)
        print("TRAIN EPOCH LOSS: {}".format(train_loss))

        # 测试
        transformer.eval()
        test_loss = 0.
        for i, (src, tgt) in enumerate(test_dataloader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                                 tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            test_loss += loss.item()
        test_loss = test_loss / len(test_dataloader)
        print("TEST EPOCH: {}  LOSS: {}".format(epoch, test_loss))

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(transformer.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    train()



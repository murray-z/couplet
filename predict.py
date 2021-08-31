# -*- coding: utf-8 -*-

import torch
from seq2seq import Encoder, Decoder, EncoderDecoder
from data_helper import get_vocab


# 加载模型
embed_size, num_hiddens, num_layers, dropout = 128, 128, 2, 0.5
batch_size, num_steps = 50, 10000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_save_path = "./model/model.pth"
word2id, id2word, vocab_size = get_vocab()
encoder = Encoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
decoder = Decoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
net.load_state_dict(torch.load(model_save_path))
net.eval()


def predict(net, src_sentence, num_steps, device):
    src_tokens = [word2id.get(w, word2id["<unk>"]) for w in (["<bos>"] + list(src_sentence.strip()) + ["<eos>"])]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs)
    dec_X = torch.unsqueeze(torch.tensor(word2id["<bos>"], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(min(len(enc_valid_len)-2, num_steps)):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == word2id["<eos>"]:
            break
        output_seq.append(pred)
    return "".join([id2word[i] for i in output_seq])











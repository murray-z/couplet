# -*- coding: utf-8 -*-

import torch
from data_helper import get_vocab
from seq2seq_transformer import generate_square_subsequent_mask
from seq2seq_transformer import Seq2SeqTransformer


def greedy_decoder(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encoder(src, src_mask)
    # decoder start <bos>
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(DEVICE)
        out = model.decoder(ys, memory, tgt_mask)
        prob = model.generator(out[:, -1, :])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src_sentence):
    model.eval()
    src = torch.LongTensor([[WORD2ID.get(w, WORD2ID["<unk>"]) for w in src_sentence]])
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decoder(model, src, src_mask, max_len=num_tokens, start_symbol=BOS_IDX).flatten()
    return "".join([ID2WORD[i.item()] for i in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


if __name__ == '__main__':
    WORD2ID, ID2WORD, VOCAB_SIZE = get_vocab()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PAD_IDX = WORD2ID["<pad>"]
    UNK_IDX = WORD2ID["<unk>"]
    EOS_IDX = WORD2ID["<eos>"]
    BOS_IDX = WORD2ID["<bos>"]
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    MODEL_SAVE_PATH = "./model/model_transformer.pth"

    transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS,
                                     num_decoder_layers=NUM_DECODER_LAYERS,
                                     embed_size=EMB_SIZE,
                                     nhead=NHEAD,
                                     vocab_size=VOCAB_SIZE,
                                     dim_feedforward=FFN_HID_DIM)

    transformer.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    transformer.to(DEVICE)

    texts = """
       风 摇 树 树 还 挺 
       愿 景 天 成 无 墨 迹 
       丹 枫 江 冷 人 初 去 
       忽 忽 几 晨 昏 ， 离 别 间 之 ， 疾 病 间 之 ， 不 及 终 年 同 静 好 
       闲 来 野 钓 人 稀 处 
       毋 人 负 我 ， 毋 我 负 人 ， 柳 下 虽 和 有 介 称 ， 先 生 字 此 ， 可 以 谥 此 
       投 石 向 天 跟 命 斗 
       深 院 落 滕 花 ， 石 不 点 头 龙 不 语 
       不 畏 鸿 门 传 汉 祚 
       新 居 落 成 创 业 始
       腾 飞 上 铁 ， 锐 意 改 革 谋 发 展 ， 勇 当 千 里 马 
       风 弦 未 拨 心 先 乱 
       花 梦 粘 于 春 袖 口 
       晋 世 文 章 昌 二 陆 
       一 句 相 思 吟 岁 月 
       几 树 梅 花 数 竿 竹 
       未 舍 东 江 开 口 咏 
       彩 屏 如 画 ， 望 秀 美 崤 函 ， 花 团 锦 簇
       """
    texts = texts.strip().split("\n")
    for text in texts:
        text = text.replace(" ", "")
        res = translate(transformer, text)
        print("{} -> {}".format(text, res))
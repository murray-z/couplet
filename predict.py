# -*- coding: utf-8 -*-

import torch
from seq2seq import Encoder, Decoder, EncoderDecoder
from data_helper import get_vocab


# 加载模型
embed_size, num_hiddens, num_layers, dropout = 128, 128, 2, 0.5
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_save_path = "./model/model.pth"
word2id, id2word, vocab_size = get_vocab()
encoder = Encoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
decoder = Decoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
net.load_state_dict(torch.load(model_save_path))
net.eval()


def predict(net, src_sentence, num_steps, device):
    src_tokens = [word2id.get(w, word2id["<unk>"]) for w in list(src_sentence.strip())]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X)
    dec_state = net.decoder.init_state(enc_outputs)
    dec_X = torch.unsqueeze(torch.tensor([word2id["<bos>"]], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(min(num_steps, len(src_sentence.strip()))):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == word2id["<eos>"]:
            break
        output_seq.append(pred)
    return "".join([id2word[i] for i in output_seq])


if __name__ == '__main__':
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
        res = predict(net, text, 30, device)
        print("{} -> {}".format(text, res))








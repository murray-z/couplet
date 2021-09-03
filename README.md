# 采用seq2seq架构训练对对联模型

- 数据
    - 数据来源：https://github.com/wb14123/couplet-dataset
    - 下载数据放在"./data"
    - 修改vocabs
      - 删除 \<s>, \</s>
      - 添加 \<pad>, \<unk>, \<bos>, \<eos>
    
- 训练
    - python seq2seq.py
    - python seq2seq_transformer.py
    



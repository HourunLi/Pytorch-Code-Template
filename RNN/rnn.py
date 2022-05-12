import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

"""
生成的诗句：
G日照香炉生，山飞玉树障。声声一已尽，林叶坠兰馨。E
G清明时节雨，饮水入前程。已远逢乡雁，何人识故乡。E
G风沙不可念，月上望苍苍。来去伊陵上，应怜故国游。E
G花开山翠干，度日白头春。别有江山老，何人识我卿。E
G雪彩生前树，山寒雪满城。何当一攀折，征路欲何依。E
G月色临城顶，山厨晚尚鸣。戴山如有意，纵酒早残机。E
G雨露随寒食，阴云蔽红井。望看烟草上，落日照波霞。E
G日暮春风起，山深万里秋。无人不得遇，应是旧平岐。E
G朝入洞庭边，夜来天籁止。月明九枝下，树下长松沙。E
G三年花落花，为我谢公丘。白首人相见，浮云不可论。E
G九日山前路，平生不自闲。江边千里外，伫有一家心。E
"""


# 古诗生成
# 主函数在main.py中
# 本py文件中的代码需要进行填空

# word embedding层
# 这里用于给每一个汉字字符(例如'床'，'前'等)以及特殊符号, 用向量进行表示


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim))  # 初始化
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embeding_random_intial))

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed

# RNN模型
# 模型可以根据当前输入的一系列词预测下一个出现的词是什么


class RNN_model(nn.Module):
    def __init__(self, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()
        self.word_embedding_lookup = word_embedding
        self.vocab_length = vocab_len  # 可选择的单词数目 或者说 word embedding层的word数目
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # 这里你需要定义 "self.rnn_lstm"
        # 其中输入特征大小是 "word_embedding_dim"
        #    输出特征大小是 "lstm_hidden_dim"
        # 这里的LSTM应该有两层，并且输入和输出的tensor都是(batch, seq, feature)大小
        # (提示：LSTM层或许torch.nn中有对应的网络层,pytorch官方文档也许有说明)
        # 填空：

        self.rnn_lstm = nn.LSTM(
            input_size=self.word_embedding_dim, hidden_size=self.lstm_dim, num_layers=2, batch_first=True)

        ##########################################
        self.fc = nn.Linear(self.lstm_dim, self.vocab_length)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, sentence, batch_size, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(
            batch_size, -1, self.word_embedding_dim)
        # print(batch_size)
        print(batch_input.shape)
        # return
        ################################################
        # 这里你需要将上面的"batch_input"输入到你在rnn模型中定义的lstm层中
        # lstm的隐藏层输出应该被定义叫做变量"output", 初始的隐藏层(initial hidden state)和记忆层(initial cell state)应该是0向量.
        # 填空
        # batch_input : (batch, seq, feature)
        # Defaults to zeros if (h_0, c_0) is not provided.
        output, (_, _) = self.rnn_lstm(batch_input)
        ################################################
        # print(output.shape)
        out = output.contiguous().view(-1, self.lstm_dim)
        # out.size: (batch_size * sequence_length ,vocab_length)
        out = self.fc(out)
        if is_test:
            # 测试阶段(或者说生成诗句阶段)使用
            print("out's shape ", out.shape)
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            # 训练阶段使用
            output = out
        return output

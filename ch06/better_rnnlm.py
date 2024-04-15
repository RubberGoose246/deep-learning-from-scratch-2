# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *  # import numpy as np
from common.base_model import BaseModel


class BetterRnnlm(BaseModel):
    '''
     LSTMレイヤを2層利用し、各層にDropoutを使うモデル
     [1]で提案されたモデルをベースとし、weight tying[2][3]を利用

     [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
     [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
     [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650, dropout_ratio=0.5):
        print('=== BetterRnnlm.__init__ BEGIN ===')

        print(f'vocab_size: {vocab_size}')
        print(f'wordvec_size: {wordvec_size}')
        print(f'hidden_size: {hidden_size}')
        print(f'dropout_ratio: {dropout_ratio}')

        V, D, H = vocab_size, wordvec_size, hidden_size

        print(f'V: {V}')
        print(f'D: {D}')
        print(f'H: {H}')

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        # hidden_size と wordvec_size を変えると形状が合わなくなり重み共有できないので、新しい重みを作成
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')

        print(f'embed_W.shape: {embed_W.shape}')
        print(f'lstm_Wx1.shape: {lstm_Wx1.shape}')
        print(f'lstm_Wh1.shape: {lstm_Wh1.shape}')
        print(f'lstm_b1.shape: {lstm_b1.shape}')
        print(f'lstm_Wx2.shape: {lstm_Wx2.shape}')
        print(f'lstm_Wh2.shape: {lstm_Wh2.shape}')
        print(f'lstm_b2.shape: {lstm_b2.shape}')
        print(f'lstm_Wh2.shape: {lstm_Wh2.shape}')
        print(f'lstm_b2.shape: {lstm_b2.shape}')
        print(f'affine_W.shape: {affine_W.shape}')
        print(f'affine_b.shape: {affine_b.shape}')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            # TimeAffine(embed_W.T, affine_b)  # weight tying!!
            TimeAffine(affine_W, affine_b)  # 重み共有ではなく、新しく作成した重みを使用
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        print(f'len(self.params): {len(self.params)}')
        for i, _ in enumerate(self.params):
          print(f'self.params[{i}].shape: {self.params[i].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        for i, _ in enumerate(self.grads):
          print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== BetterRnnlm.__init__ END ===')


    def predict(self, xs, train_flg=False):
        print('=== BetterRnnlm.predict BEGIN ===')

        print(f'xs.shape: {xs.shape}')
        print(f'train_flg: {train_flg}')

        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for i in range(len(self.layers)):
            xs = self.layers[i].forward(xs)

        print(f'xs.shape: {xs.shape}')

        print('=== BetterRnnlm.predict END ===')

        return xs

    def forward(self, xs, ts, train_flg=True):
        print('=== BetterRnnlm.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')
        print(f'ts.shape: {ts.shape}')
        print(f'train_flg: {train_flg}')

        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)

        print('=== BetterRnnlm.forward END ===')

        return loss

    def backward(self, dout=1):
        print('=== BetterRnnlm.backward BEGIN ===')

        dout = self.loss_layer.backward(dout)

        # for layer in reversed(self.layers):
        #     dout = layer.backward(dout)

        for i in range(len(self.layers)):
            dout = self.layers[len(self.layers) - 1 - i].backward(dout)

        print('=== BetterRnnlm.backward END ===')

        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

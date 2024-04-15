# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== PeekyDecoder.__init__ BEGIN ===')

        print(f'vocab_size: {vocab_size}')
        print(f'wordvec_size: {wordvec_size}')
        print(f'hidden_size: {hidden_size}')

        V, D, H = vocab_size, wordvec_size, hidden_size

        print(f'V: {V}')
        print(f'D: {D}')
        print(f'H: {H}')

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        print(f'embed_W.shape: {embed_W.shape}')
        print(f'lstm_Wx.shape: {lstm_Wx.shape}')
        print(f'lstm_Wh.shape: {lstm_Wh.shape}')
        print(f'lstm_b.shape: {lstm_b.shape}')
        print(f'affine_W.shape: {affine_W.shape}')
        print(f'affine_b.shape: {affine_b.shape}')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

        for i in range(len(self.params)):
            print(f'self.params[{i}].shape: {self.params[i].shape}')

        for i in range(len(self.grads)):
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== PeekyDecoder.__init__ END ===')

    def forward(self, xs, h):
        print('=== PeekyDecoder.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')
        print(f'h.shape: {h.shape}')

        N, T = xs.shape

        print(f'N: {N}')
        print(f'T: {T}')

        N, H = h.shape

        print(f'N: {N}')
        print(f'H: {H}')

        self.lstm.set_state(h)

        out = self.embed.forward(xs)

        print(f'out.shape: {out.shape}')

        hs = np.repeat(h, T, axis=0).reshape(N, T, H)

        print(f'hs.shape: {hs.shape}')

        out = np.concatenate((hs, out), axis=2)

        print(f'out.shape: {out.shape}')

        out = self.lstm.forward(out)

        print(f'out.shape: {out.shape}')

        out = np.concatenate((hs, out), axis=2)

        print(f'out.shape: {out.shape}')

        score = self.affine.forward(out)
        self.cache = H

        print(f'score.shape: {score.shape}')
        print(f'self.cache: {self.cache}')

        print('=== PeekyDecoder.forward END ===')

        return score

    def backward(self, dscore):
        print('=== PeekyDecoder.backward BEGIN ===')

        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)

        print('=== PeekyDecoder.backward END ===')

        return dh

    def generate(self, h, start_id, sample_size):
        print('=== PeekyDecoder.generate BEGIN ===')

        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        print('=== PeekyDecoder.generate END ===')

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== PeekySeq2seq.__init__ BEGIN ===')

        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

        print('=== PeekySeq2seq.__init__ END ===')

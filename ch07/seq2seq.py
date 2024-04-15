# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== Encoder.__init__ BEGIN ===')

        print(f'vocab_size: {vocab_size}')
        print(f'wordvec_size: {wordvec_size}')
        print(f'hidden_size: {hidden_size}')

        V, D, H = vocab_size, wordvec_size, hidden_size

        print(f'V: {V}')
        print(f'D: {D}')
        print(f'H: {H}')

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        print(f'embed_W.shape: {embed_W.shape}')
        print(f'lstm_Wx.shape: {lstm_Wx.shape}')
        print(f'lstm_Wh.shape: {lstm_Wh.shape}')
        print(f'lstm_b.shape: {lstm_b.shape}')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

        for i in range(len(self.params)):
            print(f'self.params[{i}].shape: {self.params[i].shape}')

        for i in range(len(self.grads)):
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== Encoder.__init__ END ===')

    def forward(self, xs):
        print('=== Encoder.forward BEGIN ===')

        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs

        print('=== Encoder.forward END ===')

        return hs[:, -1, :]

    def backward(self, dh):
        print('=== Encoder.backward BEGIN ===')

        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)

        print('=== Encoder.backward END ===')

        return dout


class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== Decoder.__init__ BEGIN ===')

        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        print('=== Decoder.__init__ END ===')

    def forward(self, xs, h):
        print('=== Decoder.forward BEGIN ===')

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)

        print('=== Decoder.forward END ===')

        return score

    def backward(self, dscore):
        print('=== Decoder.backward BEGIN ===')

        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh

        print('=== Decoder.backward END ===')

        return dh

    def generate(self, h, start_id, sample_size):
        print('=== Decoder.generate BEGIN ===')

        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        print('=== Decoder.generate END ===')

        return sampled


class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== Seq2seq.__init__ BEGIN ===')

        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

        print('=== Seq2seq.__init__ END ===')

    def forward(self, xs, ts):
        print('=== Seq2seq.forward BEGIN ===')

        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)

        print('=== Seq2seq.forward END ===')

        return loss

    def backward(self, dout=1):
        print('=== Seq2seq.backward BEGIN ===')

        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)

        print('=== Seq2seq.backward END ===')

        return dout

    def generate(self, xs, start_id, sample_size):
        print('=== Seq2seq.generate BEGIN ===')

        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)

        print('=== Seq2seq.generate END ===')

        return sampled

# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention


class AttentionEncoder(Encoder):
    def forward(self, xs):
        print('=== AttentionEncoder.forward BEGIN ===')

        print(f'P: xs.shape: {xs.shape}')

        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)

        print(f'xs.shape: {xs.shape}')
        print(f'hs.shape: {hs.shape}')

        print(f'R: hs.shape: {hs.shape}')

        print('=== AttentionEncoder.forward END ===')

        return hs

    def backward(self, dhs):
        print('=== AttentionEncoder.backward BEGIN ===')

        print(f'P: dhs.shape: {dhs.shape}')

        dout = self.lstm.backward(dhs)

        print(f'dout.shape: {dout.shape}')

        dout = self.embed.backward(dout)

        print(f'dout: {dout}')

        print(f'R: dout: {dout}')

        print('=== AttentionEncoder.backward END ===')

        return dout


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== AttentionDecoder.__init__ BEGIN ===')

        print(f'P: vocab_size: {vocab_size}')
        print(f'P: wordvec_size: {wordvec_size}')
        print(f'P: hidden_size: {hidden_size}')

        V, D, H = vocab_size, wordvec_size, hidden_size

        print(f'V: {V}')
        print(f'D: {D}')
        print(f'H: {H}')

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        print(f'embed_W.shape: {embed_W.shape}')
        print(f'lstm_Wx.shape: {lstm_Wx.shape}')
        print(f'lstm_Wh.shape: {lstm_Wh.shape}')
        print(f'lstm_b.shape: {lstm_b.shape}')
        print(f'affine_W.shape: {affine_W.shape}')
        print(f'affine_b.shape: {affine_b.shape}')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        print(f'len(self.params): {len(self.params)}')
        for i in range(len(self.params)):
            print(f'self.params[{i}].shape: {self.params[i].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        for i in range(len(self.grads)):
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== AttentionDecoder.__init__ END ===')

    def forward(self, xs, enc_hs):
        print('=== AttentionDecoder.forward BEGIN ===')

        print(f'P: xs.shape: {xs.shape}')
        print(f'P: enc_hs.shape: {enc_hs.shape}')

        h = enc_hs[:,-1]

        print(f'h.shape: {h.shape}')

        self.lstm.set_state(h)

        print(f'xs.shape: {xs.shape}')

        out = self.embed.forward(xs)

        print(f'out.shape: {out.shape}')

        dec_hs = self.lstm.forward(out)

        print(f'dec_hs.shape: {dec_hs.shape}')

        c = self.attention.forward(enc_hs, dec_hs)

        print(f'c.shape: {c.shape}')

        print('=== np.concatenate BEGIN ===')

        print(f'P: c.shape: {c.shape}')
        print(f'P: dec_hs.shape: {dec_hs.shape}')

        out = np.concatenate((c, dec_hs), axis=2)

        print(f'R: out.shape: {out.shape}')

        print('=== np.concatenate END ===')

        score = self.affine.forward(out)

        print(f'score.shape: {score.shape}')

        print(f'R: score.shape: {score.shape}')

        print('=== AttentionDecoder.forward END ===')

        return score

    def backward(self, dscore):
        print('=== AttentionDecoder.backward BEGIN ===')

        print(f'P: dscore.shape: {dscore.shape}')

        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        print(f'dout.shape: {dout.shape}')
        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H2: {H2}')
        print(f'H: {H}')

        dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]

        print(f'dc.shape: {dc.shape}')
        print(f'ddec_hs0.shape: {ddec_hs0.shape}')

        denc_hs, ddec_hs1 = self.attention.backward(dc)

        print(f'denc_hs.shape: {denc_hs.shape}')
        print(f'ddec_hs1.shape: {ddec_hs1.shape}')

        ddec_hs = ddec_hs0 + ddec_hs1

        print(f'ddec_hs.shape: {ddec_hs.shape}')

        dout = self.lstm.backward(ddec_hs)

        print(f'dout.shape: {dout.shape}')

        dh = self.lstm.dh

        print(f'dh.shape: {dh.shape}')

        denc_hs[:, -1] += dh

        print(f'denc_hs[:, -1].shape: {denc_hs[:, -1].shape}')

        self.embed.backward(dout)

        print(f'R: denc_hs.shape: {denc_hs.shape}')

        print('=== AttentionDecoder.backward END ===')

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        print('=== AttentionDecoder.generate BEGIN ===')

        print(f'P: enc_hs.shape: {enc_hs.shape}')
        print(f'P: start_id.shape: {start_id.shape}')
        print(f'P: sample_size: {sample_size}')

        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]

        print(f'sampled.shape: {sampled.shape}')
        print(f'sample_id.shape: {sample_id.shape}')
        print(f'h.shape: {h.shape}')

        self.lstm.set_state(h)

        #for _ in range(sample_size):
        for _ in range(1):
            x = np.array([sample_id]).reshape((1, 1))

            print(f'x.shape: {x.shape}')

            out = self.embed.forward(x)

            print(f'out.shape: {out.shape}')

            dec_hs = self.lstm.forward(out)

            print(f'dec_hs.shape: {dec_hs.shape}')

            c = self.attention.forward(enc_hs, dec_hs)

            print(f'c.shape: {c.shape}')

            out = np.concatenate((c, dec_hs), axis=2)

            print(f'out.shape: {out.shape}')

            score = self.affine.forward(out)

            print(f'score.shape: {score.shape}')

            sample_id = np.argmax(score.flatten())

            print(f'sample_id.shape: {sample_id.shape}')

            sampled.append(sample_id)

            print(f'sampled.shape: {sampled.shape}')

        print(f'R: sampled.shape: {sampled.shape}')

        print('=== AttentionDecoder.generate END ===')

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        print('=== AttentionSeq2seq.__init__ BEGIN ===')

        print(f'P: vocab_size: {vocab_size}')
        print(f'P: wordvec_size: {wordvec_size}')
        print(f'P: hidden_size: {hidden_size}')

        args = vocab_size, wordvec_size, hidden_size

        for i in range(len(args)):
            print(f'args[{i}]: {args[i]}')

        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

        print(f'len(self.params): {len(self.params)}')
        for i in range(len(self.params)):
            print(f'self.params[{i}].shape: {self.params[i].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        for i in range(len(self.grads)):
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== AttentionSeq2seq.__init__ END ===')


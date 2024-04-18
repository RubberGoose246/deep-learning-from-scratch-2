# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Softmax


class WeightSum:
    def __init__(self):
        print('=== WeightSum.__init__ BEGIN ===')

        self.params, self.grads = [], []
        self.cache = None

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.cache: {self.cache}')

        print('=== WeightSum.__init__ END ===')

    def forward(self, hs, a):
        print('=== WeightSum.forward BEGIN ===')

        print(f'P: hs.shape: {hs.shape}')
        print(f'P: a.shape: {a.shape}')

        N, T, H = hs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')

        ar = a.reshape(N, T, 1)#.repeat(T, axis=1)

        print(f'ar.shape: {ar.shape}')

        t = hs * ar

        print(f't.shape: {t.shape}')

        c = np.sum(t, axis=1)

        print(f'c.shape: {c.shape}')

        self.cache = (hs, ar)

        print(f'len(self.cache): {len(self.cache)}')
        for i in range(len(self.cache)):
            print(f'self.cache[{i}].shape: {self.cache[i].shape}')

        print(f'R: c.shape: {c.shape}')

        print('=== WeightSum.forward END ===')

        return c

    def backward(self, dc):
        print('=== WeightSum.backward BEGIN ===')

        print(f'P: dc.shape: {dc.shape}')

        hs, ar = self.cache

        print(f'hs.shape: {hs.shape}')
        print(f'ar.shape: {ar.shape}')

        N, T, H = hs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dhs = dt * hs
        dar = dt * ar
        da = np.sum(dar, axis=2)

        print(f'dt.shape: {dt.shape}')
        print(f'dhs.shape: {dhs.shape}')
        print(f'dar.shape: {dar.shape}')
        print(f'da.shape: {da.shape}')

        print(f'R: dhs.shape: {dhs.shape}')
        print(f'R: da.shape: {da.shape}')

        print('=== WeightSum.backward END ===')

        return dhs, da


class AttentionWeight:
    def __init__(self):
        print('=== AttentionWeight.__init__ BEGIN ===')

        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.cache: {self.cache}')

        print('=== AttentionWeight.__init__ END ===')

    def forward(self, hs, h):
        print('=== AttentionWeight.forward BEGIN ===')

        print(f'P: hs.shape: {hs.shape}')
        print(f'P: h.shape: {h.shape}')

        N, T, H = hs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')

        hr = h.reshape(N, 1, H)#.repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        print(f'hr.shape: {hr.shape}')
        print(f't.shape: {t.shape}')
        print(f's.shape: {s.shape}')
        print(f'a.shape: {a.shape}')

        self.cache = (hs, hr)

        print(f'len(self.cache): {len(self.cache)}')
        for i in range(len(self.cache)):
            print(f'self.cache[{i}].shape: {self.cache[i].shape}')

        print(f'R: a.shape: {a.shape}')

        print('=== AttentionWeight.forward END ===')

        return a

    def backward(self, da):
        print('=== AttentionWeight.backward BEGIN ===')

        print(f'P: da.shape: {da.shape}')

        hs, hr = self.cache
        N, T, H = hs.shape

        print(f'hs.shape: {hs.shape}')
        print(f'hr.shape: {hr.shape}')

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hs
        dhr = dt * hr
        dh = np.sum(dhr, axis=1)

        print(f'ds.shape: {ds.shape}')
        print(f'dt.shape: {dt.shape}')
        print(f'dhs.shape: {dhs.shape}')
        print(f'dhr.shape: {dhr.shape}')
        print(f'dh.shape: {dh.shape}')

        print(f'R: dhs.shape: {dhs.shape}')
        print(f'R: dh.shape: {dh.shape}')

        print('=== AttentionWeight.backward END ===')

        return dhs, dh


class Attention:
    def __init__(self):
        print('=== Attention.__init__ BEGIN ===')

        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.attention_weight: {self.attention_weight}')

        print('=== Attention.__init__ END ===')

    def forward(self, hs, h):
        print('=== Attention.forward BEGIN ===')

        print(f'P: hs.shape: {hs.shape}')
        print(f'P: h.shape: {h.shape}')

        a = self.attention_weight_layer.forward(hs, h)

        print(f'a.shape: {a.shape}')

        out = self.weight_sum_layer.forward(hs, a)

        print(f'out.shape: {out.shape}')

        self.attention_weight = a

        print(f'self.attention_weight.shape: {self.attention_weight.shape}')

        print(f'R: out.shape: {out.shape}')

        print('=== Attention.forward END ===')

        return out

    def backward(self, dout):
        print('=== Attention.backward BEGIN ===')

        print(f'P: dout.shape: {dout.shape}')

        dhs0, da = self.weight_sum_layer.backward(dout)

        print(f'dhs0.shape: {dhs0.shape}')
        print(f'da.shape: {da.shape}')

        dhs1, dh = self.attention_weight_layer.backward(da)

        print(f'dhs1.shape: {dhs1.shape}')
        print(f'dh.shape: {dh.shape}')

        dhs = dhs0 + dhs1

        print(f'dhs.shape: {dhs.shape}')

        print(f'R: dhs.shape: {dhs.shape}')
        print(f'R: dh.shape: {dhs.shape}')

        print('=== Attention.backward END ===')

        return dhs, dh


class TimeAttention:
    def __init__(self):
        print('=== TimeAttention.__init__ BEGIN ===')

        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.attention_weights: {self.attention_weights}')

        print('=== TimeAttention.__init__ END ===')

    def forward(self, hs_enc, hs_dec):
        print('=== TimeAttention.forward BEGIN ===')

        print(f'P: hs_enc.shape: {hs_enc.shape}')
        print(f'P: hs_dec.shape: {hs_dec.shape}')

        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')
        print(f'out.shape: {out.shape}')
        print(f'self.attention_weights: {self.attention_weights}')

        #for t in range(T):
        for t in range(1):
            print(f't: {t}')

            layer = Attention()

            print(f'hs_enc.shape: {hs_enc.shape}')
            print(f'hs_dec[:,{t},:].shape: {hs_dec[:,t,:].shape}')

            out[:, t, :] = layer.forward(hs_enc, hs_dec[:,t,:])

            print(f'out[:, {t}, :].shape: {out[:, t, :].shape}')

            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        print(f'len(self.attention_weights): {len(self.attention_weights)}')
        for i in range(len(self.attention_weights)):
            print(f'self.attention_weights[{i}].shape: {self.attention_weights[i].shape}')

        print(f'R: out.shape: {out.shape}')

        print('=== TimeAttention.forward END ===')

        return out

    def backward(self, dout):
        print('=== TimeAttention.backward BEGIN ===')

        print(f'P: dout.shape: {dout.shape}')

        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')
        print(f'dhs_enc: {dhs_enc}')
        print(f'dhs_dec.shape: {dhs_dec.shape}')

        #for t in range(T):
        for t in range(1):
            print(f't: {t}')

            layer = self.layers[t]

            print(f'dout[:, {t}, :].shape: {dout[:, t, :].shape}')

            dhs, dh = layer.backward(dout[:, t, :])

            print(f'dhs.shape: {dhs.shape}')
            print(f'dh.shape: {dh.shape}')

            dhs_enc += dhs

            print(f'dhs_enc.shape: {dhs_enc.shape}')

            dhs_dec[:,t,:] = dh

            print(f'dhs_dec[:,{t},:].shape: {dhs_dec[:,t,:].shape}')

        print(f'R: dhs_enc.shape: {dhs_enc.shape}')
        print(f'R: dhs_dec.shape: {dhs_dec.shape}')

        print('=== TimeAttention.backward END ===')

        return dhs_enc, dhs_dec

# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid
import sys


class RNN:
    def __init__(self, Wx, Wh, b):
        print('=== RNN.__init__ BEGIN ===')

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'len(b): {len(b)}')

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

        print(f'len(self.params): {len(self.params)}')
        print(f'self.params[0].shape (Wx): {self.params[0].shape}')
        print(f'self.params[1].shape (Wh): {self.params[1].shape}')
        print(f'len(self.params[2]) (b): {len(self.params[2])}')

        print(f'len(self.grads): {len(self.grads)}')
        print(f'self.grads[0].shape (np.zeros_like(Wx)): {self.grads[0].shape}')
        print(f'self.grads[1].shape (np.zeros_like(Wh)): {self.grads[1].shape}')
        print(f'len(self.grads[2]) (np.zeros_like(b)): {len(self.grads[2])}')

        print(f'self.cache: {self.cache}')

        print('=== RNN.__init__ END ===')

    def forward(self, x, h_prev):
        print('=== RNN.forward BEGIN ===')

        print(f'x.shape: {x.shape}')
        print(f'h_prev.shape: {h_prev.shape}')

        Wx, Wh, b = self.params

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)

        print(f't.shape: {t.shape}')
        print(f'h_next.shape: {h_next.shape}')
        print(f'len(self.cache): {len(self.cache)}')
        print(f'self.cache[0].shape: {self.cache[0].shape}')
        print(f'self.cache[1].shape: {self.cache[1].shape}')
        print(f'self.cache[2].shape: {self.cache[2].shape}')

        print('=== RNN.forward END ===')

        return h_next

    def backward(self, dh_next):
        print('=== RNN.backward BEGIN ===')

        print(f'dh_next.shape: {dh_next.shape}')

        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        print(f'x.shape: {x.shape}')
        print(f'h_prev.shape: {h_prev.shape}')
        print(f'h_next.shape: {h_next.shape}')

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        print(f'dt.shape: {dt.shape}')
        print(f'db.shape: {db.shape}')
        print(f'dWh.shape: {dWh.shape}')
        print(f'dh_prev.shape: {dh_prev.shape}')
        print(f'dWx.shape: {dWx.shape}')
        print(f'dx.shape: {dx.shape}')

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        print(f'self.grads[0].shape: {self.grads[0].shape}')
        print(f'self.grads[1].shape: {self.grads[1].shape}')
        print(f'self.grads[2].shape: {self.grads[2].shape}')

        print('=== RNN.backward END ===')

        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        print('=== TimeRNN.__init__ BEGIN ===')

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'len(b): {len(b)}')
        print(f'stateful: {stateful}')

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

        print(f'len(self.params): {len(self.params)}')
        print(f'self.params[0].shape (Wx): {self.params[0].shape}')
        print(f'self.params[1].shape (Wh): {self.params[1].shape}')
        print(f'len(self.params[2]) (b): {len(self.params[2])}')

        print(f'len(self.grads): {len(self.grads)}')
        print(f'self.grads[0].shape (np.zeros_like(Wx)): {self.grads[0].shape}')
        print(f'self.grads[1].shape (np.zeros_like(Wh)): {self.grads[1].shape}')
        print(f'len(self.grads[2]) (np.zeros_like(b)): {len(self.grads[2])}')

        print(f'self.layers: {self.layers}')
        print(f'self.h: {self.h}')
        print(f'self.dh: {self.dh}')
        print(f'self.stateful: {self.stateful}')

        print('=== TimeRNN.__init__ END ===')

    def forward(self, xs):
        print('=== TimeRNN.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')
        print(f'xs[0][0][:20]: {xs[0][0][:20]}')

        Wx, Wh, b = self.params

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        N, T, D = xs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')

        D, H = Wx.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')

        print(f'H: {H}')

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        print(f'hs.shape: {hs.shape}')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

            print(f'self.h.shape: {self.h.shape}')

        for t in range(T):
            layer = RNN(*self.params)

            print(f'xs.shape: {xs.shape}')
            print(f'xs[:, {t}, :].shape: {xs[:, t, :].shape}')

            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

            print(f'hs.shape: {hs.shape}')
            print(f'hs[:, {t}, :].shape: {hs[:, t, :].shape}')


        print('=== TimeRNN.forward END ===')

        return hs

    def backward(self, dhs):
        print('=== TimeRNN.backward BEGIN ===')

        print(f'dhs.shape: {dhs.shape}')

        Wx, Wh, b = self.params

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        N, T, H = dhs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'H: {H}')

        D, H = Wx.shape

        print(f'D: {D}')
        print(f'H: {H}')

        dxs = np.empty((N, T, D), dtype='f')

        print(f'dxs.shape: {dxs.shape}')

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]

            print(f'(dhs[:, {t}, :] + dh).shape: {(dhs[:, t, :] + dh).shape}')

            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        print(f'self.grads[0].shape: {self.grads[0].shape}')
        print(f'self.grads[1].shape: {self.grads[1].shape}')
        print(f'self.grads[2].shape: {self.grads[2].shape}')

        print(f'self.dh.shape: {self.dh.shape}')

        print(f'dxs.shape: {dxs.shape}')

        print('=== TimeRNN.backward END ===')

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class LSTM:
    def __init__(self, Wx, Wh, b):
        print('=== LSTM.__init__ BEGIN ===')

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        '''

        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
        b: バイアス（4つ分のバイアスをまとめる）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

        print(f'len(self.params): {len(self.params)}')
        for i in range(len(self.params)):
            print(f'self.params[{i}].shape: {self.params[i].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        for i in range(len(self.grads)):
            print(f'self.grads[{i}].shape: {self.grads[i].shape}')

        print('=== LSTM.__init__ END ===')

    def forward(self, x, h_prev, c_prev):
        print('=== LSTM.forward BEGIN ===')

        print(f'x.shape: {x.shape}')
        print(f'h_prev.shape: {h_prev.shape}')
        print(f'c_prev.shape: {c_prev.shape}')

        Wx, Wh, b = self.params
        N, H = h_prev.shape

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')
        print(f'N: {N}')
        print(f'H: {H}')

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        print(f'A.shape: {A.shape}')

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        print(f'f.shape: {f.shape}')
        print(f'g.shape: {g.shape}')
        print(f'i.shape: {i.shape}')
        print(f'o.shape: {o.shape}')

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        print(f'c_next.shape: {c_next.shape}')
        print(f'h_next.shape: {h_next.shape}')

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

        for i in range(len(self.cache)):
            print(f'self.cache[{i}].shape: {self.cache[i].shape}')

        print('=== LSTM.forward END ===')

        return h_next, c_next

    def backward(self, dh_next, dc_next):
        print('=== LSTM.backward BEGIN ===')

        print(f'dh_next.shape: {dh_next.shape}')
        print(f'dc_next: {dc_next}')

        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        print(f'x.shape: {x.shape}')
        print(f'h_prev.shape: {h_prev.shape}')
        print(f'c_prev.shape: {c_prev.shape}')
        print(f'i.shape: {i.shape}')
        print(f'f.shape: {f.shape}')
        print(f'g.shape: {g.shape}')
        print(f'o.shape: {o.shape}')
        print(f'c_next.shape: {c_next.shape}')

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        print(f'dx.shape: {dx.shape}')
        print(f'dh_prev.shape: {dh_prev.shape}')
        print(f'dc_prev.shape: {dc_prev.shape}')

        print('=== LSTM.backward END ===')

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        print('=== TimeLSTM.__init__ BEGIN ===')

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')
        print(f'stateful: {stateful}')

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

        print(f'self.params[0].shape: {self.params[0].shape}')
        print(f'self.params[1].shape: {self.params[1].shape}')
        print(f'self.params[2].shape: {self.params[2].shape}')

        print(f'self.grads[0].shape: {self.grads[0].shape}')
        print(f'self.grads[1].shape: {self.grads[1].shape}')
        print(f'self.grads[2].shape: {self.grads[2].shape}')

        print(f'self.h: {self.h}')
        print(f'self.c: {self.c}')
        print(f'self.dh: {self.dh}')
        print(f'self.stateful: {self.stateful}')

        print('=== TimeLSTM.__init__ END ===')

    def forward(self, xs):
        print('=== TimeLSTM.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')

        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        print(f'Wx.shape: {Wx.shape}')
        print(f'Wh.shape: {Wh.shape}')
        print(f'b.shape: {b.shape}')

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')
        print(f'H: {H}')

        print(f'hs.shape: {hs.shape}')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        print(f'self.h.shape: {self.h.shape}')
        print(f'self.c.shape: {self.c.shape}')

        for t in range(T):
            for i in range(len(self.params)):
                print(f'self.params[{i}].shape: {self.params[i].shape}')

            layer = LSTM(*self.params)

            print(f'self.h.shape: {self.h.shape}')
            print(f'self.c.shape: {self.c.shape}')
            print(f'xs[:, {t}, :].shape: {xs[:, t, :].shape}')

            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)

            print(f'self.h.shape: {self.h.shape}')
            print(f'self.c.shape: {self.c.shape}')

            hs[:, t, :] = self.h

            print(f'hs[:, {t}, :].shape: {hs[:, t, :].shape}')

            self.layers.append(layer)

        print(f'hs.shape: {hs.shape}')

        print('=== TimeLSTM.forward END ===')

        return hs

    def backward(self, dhs):
        print('=== TimeLSTM.backward BEGIN ===')

        print(f'dhs.shape: {dhs.shape}')

        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]

            print(f'(dhs[:, {t}, :] + dh).shape: {(dhs[:, t, :] + dh).shape}')
            print(f'dc: {dc}')

            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)

            print(f'dx.shape: {dx.shape}')
            print(f'dh.shape: {dh.shape}')
            print(f'dc.shape: {dc.shape}')

            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        print(f'dxs.shape: {dxs.shape}')

        print('=== TimeLSTM.backward END ===')

        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeEmbedding:
    def __init__(self, W):
        print('=== TimeEmbedding.__init__ BEGIN ===')

        print(f'W.shape: {W.shape}')

        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

        print(f'len(self.params): {len(self.params)}')
        print(f'self.params[0].shape (W): {self.params[0].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        print(f'self.grads[0].shape (np.zeros_like(W)): {self.grads[0].shape}')

        print(f'self.layers: {self.layers}')
        print(f'self.W.shape: {self.W.shape}')

        print('=== TimeEmbedding.__init__ END ===')

    def forward(self, xs):
        print('=== TimeEmbedding.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')

        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'V: {V}')
        print(f'D: {D}')
        print(f'out.shape: {out.shape}')

        for t in range(T):
            layer = Embedding(self.W)

            print(f'xs[:, {t}].shape: {xs[:, t].shape}')

            out[:, t, :] = layer.forward(xs[:, t])

            self.layers.append(layer)

            #print(f'xs[:, {t}]: {xs[:, t]}')
            #print(f'out[:, {t}, :]: {out[:, t, :]}')

        print('=== TimeEmbedding.forward END ===')

        return out

    def backward(self, dout):
        print('=== TimeEmbedding.backward BEGIN ===')

        print(f'dout.shape: {dout.shape}')

        N, T, D = dout.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad

        print(f'self.grads[0].shape: {self.grads[0].shape}')

        print('=== TimeEmbedding.backward END ===')

        return None


class TimeAffine:
    def __init__(self, W, b):
        print('=== TimeAffine.__init__ BEGIN ===')

        print(f'W.shape: {W.shape}')
        print(f'b.shape): {b.shape}')

        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

        print(f'len(self.params): {len(self.params)}')
        print(f'self.params[0].shape (W): {self.params[0].shape}')
        print(f'self.params[1].shape (b): {self.params[1].shape}')

        print(f'len(self.grads): {len(self.grads)}')
        print(f'self.grads[0].shape (np.zeros_like(W)): {self.grads[0].shape}')
        print(f'self.grads[1].shape (np.zeros_like(b)): {self.grads[1].shape}')

        print(f'self.x): {self.x}')

        print('=== TimeAffine.__init__ END ===')

    def forward(self, x):
        print('=== TimeAffine.forward BEGIN ===')
        print(f'x.shape: {x.shape}')

        N, T, D = x.shape
        W, b = self.params

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')

        print(f'W.shape: {W.shape}')
        print(f'b.shape: {b.shape}')

        rx = x.reshape(N*T, -1)

        print(f'rx.shape: {rx.shape}')

        out = np.dot(rx, W) + b
        self.x = x


        print(f'out.shape: {out.shape}')
        print(f'self.x.shape: {self.x.shape}')

        print(f'out.reshape({N}, {T}, -1).shape: {out.reshape(N, T, -1).shape}')

        print('=== TimeAffine.forward END ===')

        return out.reshape(N, T, -1)

    def backward(self, dout):
        print('=== TimeAffine.backward BEGIN ===')

        print(f'dout.shape: {dout.shape}')

        x = self.x
        N, T, D = x.shape
        W, b = self.params

        print(f'x.shape: {x.shape}')
        print(f'N: {N}')
        print(f'T: {T}')
        print(f'D: {D}')
        print(f'W.shape: {W.shape}')
        print(f'b.shape: {b.shape}')

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        print(f'x.shape: {x.shape}')

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        print(f'dW.shape: {dW.shape}')
        print(f'db.shape: {db.shape}')
        print(f'dx.shape: {dx.shape}')

        self.grads[0][...] = dW
        self.grads[1][...] = db

        print(f'self.grads[0].shape: {self.grads[0].shape}')
        print(f'self.grads[1].shape: {self.grads[1].shape}')

        print('=== TimeAffine.backward END ===')

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        print('=== TimeSoftmaxWithLoss.__init__ BEGIN ===')

        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.cache: {self.cache}')
        print(f'self.ignore_label: {self.ignore_label}')

        print('=== TimeSoftmaxWithLoss.__init__ END ===')

    def forward(self, xs, ts):
        print('=== TimeSoftmaxWithLoss.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')
        print(f'ts.shape: {ts.shape}')

        N, T, V = xs.shape

        print(f'N: {N}')
        print(f'T: {T}')
        print(f'V: {V}')

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        print(f'ts.shape: {ts.shape}')

        mask = (ts != self.ignore_label)

        print(f'mask.shape: {mask.shape}')

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        print(f'xs.shape: {xs.shape}')
        print(f'ts.shape: {ts.shape}')
        print(f'mask.shape: {mask.shape}')

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))

        print(f'len(self.cache): {len(self.cache)}')
        print(f'self.cache[0].shape: {self.cache[0].shape}')
        print(f'self.cache[1].shape: {self.cache[1].shape}')
        print(f'self.cache[2].shape: {self.cache[2].shape}')
        print(f'self.cache[3][0]: {self.cache[3][0]}')
        print(f'self.cache[3][1]: {self.cache[3][1]}')
        print(f'self.cache[3][2]: {self.cache[3][2]}')

        print(f'loss: {loss}')

        print('=== TimeSoftmaxWithLoss.forward END ===')

        return loss

    def backward(self, dout=1):
        print('=== TimeSoftmaxWithLoss.backward BEGIN ===')

        print(f'dout: {dout}')

        ts, ys, mask, (N, T, V) = self.cache

        print(f'ts.shape: {ts.shape}')
        print(f'ys.shape: {ys.shape}')
        print(f'mask.shape: {mask.shape}')
        print(f'N: {N}')
        print(f'T: {T}')
        print(f'V: {V}')

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        print(f'mask.shape: {mask.shape}')
        #print(f'mask: {mask}')

        print(f'mask[:, np.newaxis].shape: {mask[:, np.newaxis].shape}')
        #print(f'mask[:, np.newaxis]: {mask[:, np.newaxis]}')

        print(f'dx.shape: {dx.shape}')

        dx = dx.reshape((N, T, V))

        print(f'dx.shape: {dx.shape}')

        print('=== TimeSoftmaxWithLoss.backward END ===')

        return dx


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        print('=== TimeDropout.__init__ BEGIN ===')

        print(f'dropout_ratio: {dropout_ratio}')

        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

        print(f'self.params: {self.params}')
        print(f'self.grads: {self.grads}')
        print(f'self.dropout_ratio: {self.dropout_ratio}')
        print(f'self.mask: {self.mask}')
        print(f'self.train_flg: {self.train_flg}')

        print('=== TimeDropout.__init__ END ===')

    def forward(self, xs):
        print('=== TimeDropout.forward BEGIN ===')

        print(f'xs.shape: {xs.shape}')

        print(f'self.train_flg: {self.train_flg}')

        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            print(f'(xs * self.mask).shape: {(xs * self.mask).shape}')

            print('=== TimeDropout.forward END 1 ===')

            return xs * self.mask
        else:
            print('=== TimeDropout.forward END 2 ===')

            print(f'xs.shape: {xs.shape}')

            return xs

    def backward(self, dout):
        print('=== TimeDropout.backward BEGIN ===')

        print(f'dout.shape: {dout.shape}')

        print('=== TimeDropout.backward END ===')

        return dout * self.mask


class TimeBiLSTM:
    def __init__(self, Wx1, Wh1, b1,
                 Wx2, Wh2, b2, stateful=False):
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs):
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out

    def backward(self, dhs):
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs

# ====================================================================== #
# 以下に示すレイヤは、本書で説明をおこなっていないレイヤの実装もしくは
# 処理速度よりも分かりやすさを優先したレイヤの実装です。
#
# TimeSigmoidWithLoss: 時系列データのためのシグモイド損失レイヤ
# GRU: GRUレイヤ
# TimeGRU: 時系列データのためのGRUレイヤ
# BiTimeLSTM: 双方向LSTMレイヤ
# Simple_TimeSoftmaxWithLoss：単純なTimeSoftmaxWithLossレイヤの実装
# Simple_TimeAffine: 単純なTimeAffineレイヤの実装
# ====================================================================== #


class TimeSigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


class GRU:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（3つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（3つ分の重みをまとめる）
        b: バイアス（3つ分のバイアスをまとめる）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bh)
        h_next = (1-z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat =dh_next * z
        dh_prev = dh_next * (1-z)

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dbh = np.sum(dt, axis=0)
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1-z)
        dbz = np.sum(dt, axis=0)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev


class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Simple_TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss

    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class Simple_TimeAffine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype='f')
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype='f')
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs





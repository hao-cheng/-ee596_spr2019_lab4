import numpy as np

DTYPE = np.double


def sigmoid(x):
    return (np.tanh(0.5 * x) + 1) * 0.5


class RnnUnit:
    """This class stores the activations of the RNN Unit"""

    def __init__(self, hidden_size, batch_size, dtype):
        self.h = np.zeros([hidden_size, batch_size], dtype=dtype)

    def forward_function(self, x, hprev, Whx, Whh):
        """Returns h = sigmoid(Whx*x + Whh*h)"""
        # =========================
        # TODO: finish the codes here
        # self.h = ...
        raise NotImplementedError()
        # =========================
        h = np.copy(self.h)
        return h

    def backward_function(self, x, hprev, dEdh, Whx, Whh):
        """Returns dEdhprev, dEdWhx, and dEdWhh"""
        # =========================
        # TODO: finish the codes here
        # dy = ...
        # dEdhprev = ...
        # dEdWhx = ...
        # dEdWhh = ...
        raise NotImplementedError()
        # =========================

        return dEdhprev, dEdWhx, dEdWhh


def test_rnn_unit():
    """Tests for the gradient computation of the single RNNUnit"""
    hidden_size = 10
    input_size = 5
    batch_size = 2
    Whh = np.zeros([hidden_size, hidden_size], dtype=DTYPE)
    Whh += np.random.uniform(-0.1, 0.1, [hidden_size, hidden_size])
    Whx = np.zeros([hidden_size, input_size], dtype=DTYPE)
    Whx += np.random.uniform(-0.1, 0.1, [hidden_size, input_size])
    x = np.zeros([input_size, batch_size], dtype=DTYPE)
    x += np.random.uniform(-0.1, 0.1, [input_size, batch_size])
    hprev = np.zeros([hidden_size, batch_size], dtype=DTYPE)
    hprev += np.random.uniform(-0.1, 0.1, [hidden_size, batch_size])

    rnn_unit = RnnUnit(hidden_size, batch_size, DTYPE)

    # Exact gradient computation
    h = rnn_unit.forward_function(x, hprev, Whx, Whh)
    E = np.sum(h)
    dEdh = np.ones([hidden_size, batch_size], dtype=DTYPE)
    dEdhprev, dWhx, dWhh = rnn_unit.backward_function(x, hprev, dEdh, Whx, Whh)

    # Numerical gradient computation
    epsilon = 1e-7
    numdWhh = np.zeros([hidden_size, hidden_size], dtype=DTYPE)
    for i in range(hidden_size):
        for j in range(hidden_size):
            newWhh = np.copy(Whh)
            newWhh[i, j] += epsilon

            h = rnn_unit.forward_function(x, hprev, Whx, newWhh)
            newE = np.sum(h)
            numdWhh[i, j] = (newE - E) / epsilon

    diff = abs(np.sum(numdWhh - dWhh))
    assert diff < 1e-4
    print('dWhh test passed! abs(expected - actual) =', diff)

    # Numerical gradient computation
    epsilon = 1e-7
    numdWhx = np.zeros([hidden_size, input_size], dtype=DTYPE)
    for i in range(hidden_size):
        for j in range(input_size):
            newWhx = np.copy(Whx)
            newWhx[i, j] += epsilon

            h = rnn_unit.forward_function(x, hprev, newWhx, Whh)
            newE = np.sum(h)
            numdWhx[i, j] = (newE - E) / epsilon

    diff = abs(np.sum(numdWhx - dWhx))
    assert diff < 1e-4
    print('dWhx test passed! abs(expected - actual) =', diff)


if __name__ == '__main__':
    test_rnn_unit()

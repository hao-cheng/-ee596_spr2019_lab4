import os
import sys
import pickle

import numpy as np

from .rnn_unit import RnnUnit
from .softmax_unit import SoftmaxUnit

DTYPE = np.double


class RnnModel:
    """An RNN model with an softmax output layer."""

    def __init__(self,
                 init_range=0.1,
                 learning_rate=0.1,
                 verbose=True,
                 batch_size=1,
                 bptt_unfold_level=1,
                 input_size=0,
                 hidden_size=0,
                 output_size=0):
        # optimization parameters
        self.init_range = init_range
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.batch_size = batch_size

        # neuralnet structure params
        self.bptt_unfold_level = bptt_unfold_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_units = []
        self.softmax_units = []

        self.Whx = None
        self.Whh = None
        self.Woh = None
        self.bo = None
        self.last_Whx = None
        self.last_Whh = None
        self.last_Woh = None
        self.last_bo = None
        self.hprev = None

    def initialize_parameters(self):
        """Randomly initializes the connection weights (bias stays at 0)"""
        self.Whx += np.random.uniform(-self.init_range, self.init_range, self.Whx.shape)
        self.Whh += np.random.uniform(-self.init_range, self.init_range, self.Whh.shape)
        self.Woh += np.random.uniform(-self.init_range, self.init_range, self.Woh.shape)

    def reset_states(self, idxs=None):
        if idxs is None:
            idxs = list(range(self.hprev.shape[1]))
        self.hprev[:, idxs] = 0

    def allocate_model(self):
        """Allocates model parameters and placeholders"""
        # Allocate model parameters
        self.Whx = np.zeros([self.hidden_size, self.input_size], dtype=DTYPE)
        self.Whh = np.zeros([self.hidden_size, self.hidden_size], dtype=DTYPE)
        self.Woh = np.zeros([self.output_size, self.hidden_size], dtype=DTYPE)
        self.bo = np.zeros([self.output_size, 1], dtype=DTYPE)

        self.last_Whx = np.zeros([self.hidden_size, self.input_size], dtype=DTYPE)
        self.last_Whh = np.zeros([self.hidden_size, self.hidden_size], dtype=DTYPE)
        self.last_Woh = np.zeros([self.output_size, self.hidden_size], dtype=DTYPE)
        self.last_bo = np.zeros([self.output_size, 1], dtype=DTYPE)

        # Allocate states
        self.hprev = np.zeros([self.hidden_size, self.batch_size], dtype=DTYPE)

        # Allocate activations and softmax
        for _ in range(self.bptt_unfold_level):
            self.rnn_units.append(RnnUnit(self.hidden_size, self.batch_size, DTYPE))
            self.softmax_units.append(SoftmaxUnit(self.output_size, self.batch_size, DTYPE))

    def read_model(self, fname, eval=True):
        """Reads model from file"""
        if not os.path.exists(fname):
            print(
                'Error: Model file {} does not exist!\n'.format(fname),
                file=sys.stderr
            )
            exit(1)

        with open(fname, 'rb') as fin:
            model = pickle.load(fin)
            print('=========Reading Model========')
            self.init_range = model['init_range']
            self.input_size = model['input_size']
            self.hidden_size = model['hidden_size']
            self.output_size = model['output_size']
            self.learning_rate = model['learning_rate']
            if eval:
                self.bptt_unfold_level = 1
                self.batch_size = 1
            else:
                self.bptt_unfold_level = model['bptt_unfold_level']

            self.allocate_model()

            self.Whx = model['Whx']
            self.Whh = model['Whh']
            self.Woh = model['Woh']
            self.bo = model['bo']
            print('=========Reading Done========')

    def write_model(self, fname):
        """Writes model to file"""
        model = dict()
        model['init_range'] = self.init_range
        model['input_size'] = self.input_size
        model['hidden_size'] = self.hidden_size
        model['output_size'] = self.output_size
        model['learning_rate'] = self.learning_rate
        model['bptt_unfold_level'] = self.bptt_unfold_level

        model['Whx'] = self.Whx
        model['Whh'] = self.Whh
        model['Woh'] = self.Woh
        model['bo'] = self.bo

        with open(fname, 'wb') as fout:
            print('=========Writing Model========')
            pickle.dump(model, fout)
            print('=========Writing Done========')

    def forward_propagate(self, input_idxs, target_idxs):
        loss = 0
        probs = []

        for i, (input_idx, target_idx) in enumerate(zip(input_idxs, target_idxs)):
            assert len(input_idx) == self.batch_size
            assert len(target_idx) == 2
            assert len(target_idx[0]) == self.batch_size
            assert len(target_idx[1]) == self.batch_size
            x = np.zeros([self.input_size, self.batch_size], dtype=DTYPE)
            x[input_idx, list(range(self.batch_size))] = 1.0

            # =========================
            # TODO: finish the codes here
            # h = ...
            raise NotImplementedError()
            # =========================

            p = self.softmax_units[i].forward_function(h, self.Woh, self.bo)
            probs += [p]
            loss += self.softmax_units[i].compute_loss(target_idx)
            self.hprev = h
        return loss, probs

    def backward_propagate(self, input_idxs, target_idxs):
        dWhh = np.zeros(self.Whh.shape)
        dWoh = np.zeros(self.Woh.shape)
        dWhx = np.zeros(self.Whx.shape)
        dbo = np.zeros(self.bo.shape)
        dEdh = np.zeros([self.hidden_size, self.batch_size])
        for i in range(self.bptt_unfold_level-1, -1, -1):
            target_idx = target_idxs[i]
            input_idx = input_idxs[i]
            # Retrieve activations
            h = self.rnn_units[i].h
            if i > 0:
                hprev = self.rnn_units[i-1].h
            else:
                hprev = np.zeros([self.hidden_size, self.batch_size])
            # Backprop the Softmax
            (
                dEdh_softmax, l_dWoh, l_dbo
            ) = self.softmax_units[i].backward_function(
                target_idx, h, self.Woh, self.bo)

            # Backprop the RNN
            x = np.zeros([self.input_size, self.batch_size], dtype=DTYPE)
            x[input_idx, list(range(self.batch_size))] = 1.0

            # =========================
            # TODO: finish the codes here
            # dEdhprev, l_dWhx, l_dWhh = ...
            raise NotImplementedError()
            # =========================

            # Update the gradient accumulators
            dEdh = dEdhprev
            dWhh += l_dWhh
            dWoh += l_dWoh
            dWhx += l_dWhx
            dbo += l_dbo
        return dWhh, dWoh, dWhx, dbo

    def update_weight(self, dWhh, dWoh, dWhx, dbo):
        dWhh *= self.learning_rate
        dWoh *= self.learning_rate
        dWhx *= self.learning_rate
        dbo *= self.learning_rate
        self.Whh += dWhh
        self.Woh += dWoh
        self.Whx += dWhx
        self.bo += dbo

    def restore_model(self):
        self.Whh[:] = self.last_Whh
        self.Woh[:] = self.last_Woh
        self.Whx[:] = self.last_Whx
        self.bo[:] = self.last_bo

    def cache_model(self):
        self.last_Whh[:] = self.Whh
        self.last_Woh[:] = self.Woh
        self.last_Whx[:] = self.Whx
        self.last_bo[:] = self.bo


def test_rnn_model():
    """Tests for the gradient computation of the whole RNN"""
    rnn_model = RnnModel(
        batch_size=3,
        bptt_unfold_level=10,
        hidden_size=20,
        input_size=5,
        output_size=15,
        init_range=0.1,
        learning_rate=0.1
    )
    rnn_model.allocate_model()
    rnn_model.initialize_parameters()

    # Fake indices
    input_idxs = []
    target_idxs = []
    for t in range(rnn_model.bptt_unfold_level):
        input_idx = []
        target_idx = []
        target_mult = []
        for b in range(rnn_model.batch_size):
            input_ind = np.random.randint(0, rnn_model.input_size)
            input_idx.append(input_ind)
            target_ind = np.random.randint(0, rnn_model.output_size)
            target_idx.append(target_ind)
            target_mult.append(1.0)
        input_idxs.append(input_idx)
        target_idxs.append((target_idx, target_mult))

    # print(input_idxs)
    # print(target_idxs)

    # Numerical gradient computation for Woh
    rnn_model.reset_states()
    E, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx, dbo = rnn_model.backward_propagate(input_idxs, target_idxs)

    epsilon = 1e-7
    baseWoh = np.copy(rnn_model.Woh)
    numdWoh = np.zeros([rnn_model.output_size, rnn_model.hidden_size], dtype=DTYPE)
    for i in range(rnn_model.output_size):
        for j in range(rnn_model.hidden_size):
            newWoh = np.copy(baseWoh)
            newWoh[i, j] += epsilon
            rnn_model.Woh = newWoh

            rnn_model.reset_states()
            newE, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
            numdWoh[i, j] = (newE - E) / epsilon

    diff = abs(np.sum(numdWoh - dWoh))
    assert diff < 1e-4
    print('dWoh test passed! abs(expected - actual) =', diff)

    # Numerical gradient computation for dbo
    rnn_model.reset_states()
    E, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx, dbo = rnn_model.backward_propagate(input_idxs, target_idxs)

    epsilon = 1e-7
    basebo = np.copy(rnn_model.bo)
    numdbo = np.zeros([rnn_model.output_size, 1], dtype=DTYPE)
    for i in range(rnn_model.output_size):
        newbo = np.copy(basebo)
        newbo[i] += epsilon
        rnn_model.bo = newbo

        rnn_model.reset_states()
        newE, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
        numdbo[i] = (newE - E) / epsilon

    diff = abs(np.sum(numdbo - dbo))
    assert diff < 1e-4
    print('dbo test passed! abs(expected - actual) =', diff)

    # Numerical gradient computation for Whx
    rnn_model.reset_states()
    E, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx, dbo = rnn_model.backward_propagate(input_idxs, target_idxs)

    epsilon = 1e-7
    baseWhx = np.copy(rnn_model.Whx)
    numdWhx = np.zeros([rnn_model.hidden_size, rnn_model.input_size], dtype=DTYPE)
    for i in range(rnn_model.hidden_size):
        for j in range(rnn_model.input_size):
            newWhx = np.copy(baseWhx)
            newWhx[i, j] += epsilon
            rnn_model.Whx = newWhx

            rnn_model.reset_states()
            newE, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
            numdWhx[i, j] = (newE - E) / epsilon

    diff = abs(np.sum(numdWhx - dWhx))
    assert diff < 1e-4
    print('dWhx test passed! abs(expected - actual) =', diff)

    # Numerical gradient computation for Whh
    rnn_model.reset_states()
    E, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
    dWhh, dWoh, dWhx, dbo = rnn_model.backward_propagate(input_idxs, target_idxs)

    epsilon = 1e-7
    baseWhh = np.copy(rnn_model.Whh)
    numdWhh = np.zeros([rnn_model.hidden_size, rnn_model.hidden_size], dtype=DTYPE)
    for i in range(rnn_model.hidden_size):
        for j in range(rnn_model.hidden_size):
            newWhh = np.copy(baseWhh)
            newWhh[i, j] += epsilon
            rnn_model.Whh = newWhh

            rnn_model.reset_states()
            newE, _ = rnn_model.forward_propagate(input_idxs, target_idxs)
            numdWhh[i, j] = (newE - E) / epsilon

    diff = abs(np.sum(numdWhh - dWhh))
    assert diff < 1e-4
    print('dWhh test passed! abs(expected - actual) =', diff)


if __name__ == '__main__':
    test_rnn_model()

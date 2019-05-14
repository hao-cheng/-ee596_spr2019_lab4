import sys
import argparse

import numpy as np
from .rnn_model import RnnModel


def build_vocab(fn):
    number4idx = {}
    idx4number ={}
    number_idx = 0
    with open(fn) as fin:
        for line in fin:
            number = line.strip()
            idx4number[number] = number_idx
            number4idx[number_idx] = number
            number_idx += 1
    if '</s>' not in idx4number:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in idx4number:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return idx4number, number4idx


def sort_number_sequence(number_seq_str, model, number4idx, idx4number):
    """Sorts a number sequence using a trained RNN model."""
    model.reset_states()

    # NOTE: we use a dummy target_idx since we don't use the loss computed in the `forward_propagate` method.
    dummy_target_idx = [([0], [0.0])]

    # Reads in the input number sequence
    nums = number_seq_str.split(' ')
    nums.insert(0, '</s>')
    for num in nums:
        idx = idx4number[num]
        input_idx = [[idx]]
        _, probs = model.forward_propagate(input_idx, dummy_target_idx)

    # Greedily searches for the most likely number
    sorted_nums = []
    max_len = 5
    curr_num = '<sort>'
    for _ in range(max_len):
        input_idx = [[idx4number[curr_num]]]
        # =========================
        # TODO: finish the codes here
        # probs = ...
        # curr_num = ...
        # sorted_nums.append(...)
        raise NotImplementedError()
        # =========================

    return ' '.join(sorted_nums)


def main():
    pa = argparse.ArgumentParser(description='rnn a trained RNN sequence model for sorting number')
    pa.add_argument('--vocabfile',
                    required=True,
                    help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel',
                    required=True,
                    help='inmodel name (REQUIRED)')
    args = pa.parse_args()

    idx4number, number4idx = build_vocab(args.vocabfile)

    rnn_model = RnnModel()
    rnn_model.read_model(args.inmodel)

    while True:
        number_seq_str = input('Enter 5 unique integers in [0, 9] separated by spaces: ')
        sorted_number_seq_str = sort_number_sequence(number_seq_str, rnn_model, number4idx, idx4number)
        print('The sorted numbers in ascending order is: {}'.format(sorted_number_seq_str))


if __name__ == '__main__':
    main()

import sys
import argparse

import numpy as np

from .rnn_model import RnnModel


def build_vocab(fn):
    char4idx = {}
    idx4char ={}
    char_idx = 0
    with open(fn) as fin:
        for line in fin:
            char = line.strip()
            idx4char[char] = char_idx
            char4idx[char_idx] = char
            char_idx += 1
    if '</s>' not in idx4char:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in idx4char:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return idx4char, char4idx


def generate_sentence(first_word, model, char4idx, idx4char):
    """Generates a sentence given the first word."""
    # NOTE: we use a dummy target_idx since we don't use the loss computed in the `forward_propagate` method.
    dummy_target_idx = [([0], [0.0])]

    # Reads in the initial character sequence
    model.reset_states()
    characters = ' '.join(first_word).split()
    characters.insert(0, '</s>')
    for character in characters:
        input_idx = [[idx4char[character]]]
        _, probs = model.forward_propagate(input_idx, dummy_target_idx)

    # Randomly sample the remaining characters until the end-of-sentence token </s> is generated
    # or maximum number of words (including <sep>) is reached
    max_num_words = 60
    character_vocab = [char4idx[idx] for idx in range(len(char4idx))]
    curr_word = []
    curr_character = '<sep>'
    words = [first_word]
    while len(words) <= max_num_words:
        input_idx = [[idx4char[curr_character]]]

        # ======================
        # TODO: finish the codes here
        # probs = ...
        # curr_character = np.random.choice(...)
        raise NotImplementedError()
        # ======================

        if curr_character == '</s>':
            words.append(''.join(curr_word))
            break
        if curr_character == '<sep>':
            words.append(''.join(curr_word))
            curr_word = []
        else:
            curr_word.append(curr_character)

    return ' '.join(words)


def main():
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    pa.add_argument('--vocabfile',
                    required=True,
                    help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel',
                    required=True,
                    help='inmodel name (REQUIRED)')
    pa.add_argument('--sample-sent', action='store_true',
                    dest='sample_sent', help='sample LM to generate sentence')
    pa.set_defaults(sample_sent=False)
    args = pa.parse_args()

    idx4char, char4idx = build_vocab(args.vocabfile)

    rnn_model = RnnModel()
    rnn_model.read_model(args.inmodel)

    while True:
        first_word = input('Enter a single word (lowercased) to start with: ')
        sent = generate_sentence(first_word, rnn_model, char4idx, idx4char)
        print('The sampled sentence starting with "{}" is: {}'.format(first_word, sent))


if __name__ == '__main__':
    main()

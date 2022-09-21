# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:26 PM


from collections import Counter, OrderedDict
from pathlib import Path
from typing import Union

from torchtext.vocab import vocab, Vocab

from . import entities_list


def entities2iob_labels(entities: list):
    '''
    get all iob string label by entities
    :param entities:
    :return:
    '''
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
    tags.append('O')
    return tags


def create_sorted_ordereddict(counter: Counter) -> OrderedDict:
    # For reproduce same behavior of torchtext 0.6.0
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    counter_dict = OrderedDict()
    for k, c in words_and_frequencies:
        counter_dict[k] = c
    return counter_dict


def create_vocab(counter: Union[OrderedDict, Counter]) -> Vocab:
    sorted_counter = create_sorted_ordereddict(counter)
    v = vocab(sorted_counter, specials=['<pad>', '<unk>'], special_first=False, min_freq=1)
    v.set_default_index(v['<unk>'])
    return v


with open(Path(__file__).parent.joinpath('keys.txt'), 'r', encoding='utf-8') as fin:
    keys_vocab_counter = Counter(list(fin.read().strip()))

keys_vocab = create_vocab(keys_vocab_counter)
iob_labels_vocab = create_vocab(create_sorted_ordereddict(Counter(entities2iob_labels(entities_list.Entities_list))))
entities_vocab = create_vocab(create_sorted_ordereddict(Counter(entities_list.Entities_list)))

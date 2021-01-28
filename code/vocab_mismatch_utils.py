from __future__ import absolute_import, division, print_function

from constants import *

import random
import collections
import unicodedata

import six

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_bert_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0 # 0 is reserved for padding
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def utf8len(s):
    return len(s.encode('utf-8'))

def mismatch_vocab_random_by_categories(vocab):
    vocab_inverse = collections.OrderedDict()
    for k, v in vocab.items():
        vocab_inverse[v] = k

    # special tokens
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']
    special_tokens_ids = [vocab[t] for t in special_tokens]

    # [unused.*] mismatch
    unused_mismatch = dict()
    unused_original = [_id for _id in range(999) if _id not in special_tokens_ids]
    unused_permute = [_id for _id in range(999) if _id not in special_tokens_ids]
    random.shuffle(unused_permute)
    for i in range(len(unused_original)):
        unused_mismatch[unused_original[i]] = unused_permute[i]

    # special char mismatch
    def special_range_mismatch(lower_id_b, upper_id_b):
        special_char_mismatch = dict()
        special_char_original = [_id for _id in range(lower_id_b, upper_id_b)]
        special_char_permute = [_id for _id in range(lower_id_b, upper_id_b)]
        random.shuffle(special_char_permute)
        for i in range(len(special_char_original)):
            special_char_mismatch[special_char_original[i]] = special_char_permute[i]
        return special_char_mismatch

    special_char_mismatch_1 = special_range_mismatch(999, 1014)
    special_char_mismatch_2 = special_range_mismatch(1014, 1024)
    special_char_mismatch_3 = special_range_mismatch(1024, 1037)
    special_char_mismatch_4 = special_range_mismatch(1037, 1063)
    special_char_mismatch_5 = special_range_mismatch(1063, 1996)

    # by bytes mismatch
    by_byte_mismatch = dict()
    by_byte_category = dict()
    for v, id in vocab.items():
        if id > 1995:
            v_len = utf8len(v)
            if v_len in by_byte_category.keys():
                by_byte_category[v_len].append(id)
            else:
                by_byte_category[v_len] = [id]
    by_byte_category_permute = dict()
    for k, v in by_byte_category.items():
        by_byte_category_permute[k] = random.sample(v, len(v))
    for k, v in by_byte_category_permute.items():
        original_v = by_byte_category[k]
        assert len(v) == len(original_v)
        for i in range(len(v)):
            by_byte_mismatch[original_v[i]] = v[i]

    mismatch_vocab = dict()
    for v, id in vocab.items():
        if v in special_tokens:
            mismatch_vocab[id] = id
        else:
            if id <= 998: # [unused.*] cases
                assert v.startswith("[unused")
                mismatch_vocab[id] = unused_mismatch[id]
            elif 998 < id and id <= 1013:
                mismatch_vocab[id] = special_char_mismatch_1[id]
            elif 1013 < id and id <= 1023:
                mismatch_vocab[id] = special_char_mismatch_2[id]
            elif 1023 < id and id <= 1036:
                mismatch_vocab[id] = special_char_mismatch_3[id]
            elif 1036 < id and id <= 1062:
                mismatch_vocab[id] = special_char_mismatch_4[id]
            elif 1062 < id and id <= 1995: 
                mismatch_vocab[id] = special_char_mismatch_5[id]
            else:
                mismatch_vocab[id] = by_byte_mismatch[id]
                
    new_vocab_str = []
    for k, v in vocab.items():
        new_vocab_str.append(vocab_inverse[mismatch_vocab[v]])
        
    assert len(new_vocab_str) == len(vocab)
    return new_vocab_str

def write_bert_vocab(vocab_list, path):
    with open(path, "w") as f:
        for w in vocab_list:
            f.write(w)
            f.write("\n")
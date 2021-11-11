from __future__ import absolute_import, division, print_function

from constants import *

import random
import collections
import unicodedata

import six

from vocab_mismatch_utils import *
from data_formatter_utils import *
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset
import transformers
import pandas as pd

import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace

import torch
import numpy as np
import copy
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
from word_forms.word_forms import get_word_forms


# translator to try it out!
def corrupt_translator(in_string, tokenizer, vocab_match):
    tokens = tokenizer.tokenize(in_string)
    translate_tokens = []
    for t in tokens:
        if t in vocab_match.keys():
            translate_tokens += [vocab_match[t]]
        else:
            translate_tokens += [t]
    out_string = " ".join(translate_tokens).replace(" ##", "").strip()
    return out_string

def plot_dist(vocab, map1, map2):
    freq_diff = []
    for k, v in vocab.items():
        diff = abs(map1[k] - map2[v])
        # print(diff)
        freq_diff.append(diff)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    g = ax.hist(freq_diff, bins=100, facecolor='r')
    plt.grid(True)
    plt.grid(color='black', linestyle='-.')
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

def token_lemma_mapping(word_dict):
    token_lemma_map = {}
    for k, v in word_dict.items():
        external_forms = get_word_forms(k)
        all_lemmas = set([])
        for e_k, e_v in external_forms.items():
            all_lemmas = all_lemmas.union(e_v)
        token_lemma_map[k] = all_lemmas
    return token_lemma_map

def rotate(list_in, k):
    list_out = copy.deepcopy(list_in)
    # speed up the rotation
    k %= len(list_out)

    for i in range(k):
        previous = list_out[-1]
        for j in range(len(list_out)):
            list_out[j], previous = previous, list_out[j]
    return list_out

def generate_vocab_match_v1(token_by_length):
    vocab_match = {}
    for _, tokens in token_by_length.items():
        tokens_copy = copy.deepcopy(tokens)
        random.shuffle(tokens_copy)
        if len(tokens_copy) >= 2:
            rotate_degree = random.randint(1, len(tokens_copy)-1)
        else:
            rotate_degree = 0
        rotated_tokens = rotate(tokens_copy, rotate_degree)
        for i in range(len(tokens)):
            vocab_match[tokens_copy[i]] = rotated_tokens[i]
    return vocab_match

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    norm_dist = distances[-1]/max(len(s1), len(s2))
    return norm_dist
    
def levenshtein_distance_mapping(token_frequency_map):
    levenshtein_distance_map = {}
    tokens = []
    for k, v in token_frequency_map.items():
        tokens.append(k)
    for i in range(0, len(tokens)):
        for j in range(i, len(tokens)):
            d = levenshteinDistance(tokens[i], tokens[j])
            levenshtein_distance_map[(tokens[i],tokens[j])] = d
            levenshtein_distance_map[(tokens[j],tokens[i])] = d
    return levenshtein_distance_map

def get_frequency_buckets(token_frequency_map):

    frequency_buckets = {}
    for k, v in token_frequency_map.items():
        if v in frequency_buckets.keys():
            frequency_buckets[v].append(k)
        else:
            frequency_buckets[v] = [k]
    frequency_buckets = collections.OrderedDict(sorted(frequency_buckets.items()))

    coarse_frequency_buckets = {}
    buffer_tokens = []
    for freq, tokens in frequency_buckets.items():
        if len(tokens) >= 10:
            curr_tokens = []
            if len(buffer_tokens) != 0:
                curr_tokens += buffer_tokens
            curr_tokens += tokens
            tokens_copy = copy.deepcopy(curr_tokens)
            coarse_frequency_buckets[len(coarse_frequency_buckets)] = tokens_copy
            buffer_tokens = [] # empty it just in case
        else:
            buffer_tokens += tokens
            if len(buffer_tokens) >= 10:
                curr_tokens = []
                curr_tokens += buffer_tokens
                tokens_copy = copy.deepcopy(curr_tokens)
                coarse_frequency_buckets[len(coarse_frequency_buckets)] = tokens_copy
                buffer_tokens = [] # empty it just in case
    if len(buffer_tokens) != 0:
        if len(coarse_frequency_buckets)-1 >= 0:
            coarse_frequency_buckets[len(coarse_frequency_buckets)-1] += buffer_tokens
        else:
            coarse_frequency_buckets[len(coarse_frequency_buckets)] = buffer_tokens
    buffer_tokens = []
    
    return coarse_frequency_buckets

def match_by_lemma_and_edit_dist(tokens, token_lemma_map):
    vocab_match = {}
    tokens_copy = copy.deepcopy(tokens)
    random.shuffle(tokens_copy)
    if len(tokens_copy) >= 2:
        rotate_degree = int(len(tokens_copy)/2)
    else:
        rotate_degree = 0
    tokens_copy_rotate = copy.deepcopy(tokens_copy)
    tokens_copy_rotate = rotate(tokens_copy_rotate, rotate_degree)
    for i in range(len(tokens)):
        vocab_match[tokens_copy[i]] = tokens_copy_rotate[i]
    # then we check their lemma and edit distance
    # in case fault case, we swap
    adjust_match = []
    for word_i, word_j in vocab_match.items():
        lemma_i = token_lemma_map[word_i]
        lemma_j = token_lemma_map[word_j]
        lemma_intersect = len(list(lemma_i.intersection(lemma_j)))
        if lemma_intersect == 0 and levenshteinDistance(word_i, word_j) > 0.3:
            continue
        else:
            adjust_match.append((word_i, word_j))
    for a_m in adjust_match:
        for word_i, word_j in vocab_match.items():
            lemma_i = token_lemma_map[a_m[0]]
            lemma_j = token_lemma_map[word_i]
            lemma_intersect = len(list(lemma_i.intersection(lemma_j)))
            if lemma_intersect == 0 and levenshteinDistance(a_m[0], word_i) > 0.3:
                # swap
                vocab_match[word_i] = a_m[1]
                vocab_match[a_m[0]] = word_j
                break
    
    return vocab_match

def generate_vocab_match_v2(token_by_length, token_frequency_map, token_lemma_map):
    vocab_match = {}
    for length, tokens in token_by_length.items():
        print(f"examining length = {length}, candidates length = {len(tokens)}")
        sub_token_frequency_map = {}
        for token in token_by_length[length]:
            sub_token_frequency_map[token] = token_frequency_map[token]
        frequency_buckets = get_frequency_buckets(sub_token_frequency_map)
        for bucket_id, tokens in frequency_buckets.items():
            sub_vocab_match = match_by_lemma_and_edit_dist(tokens, token_lemma_map)
            for k, v in sub_vocab_match.items():
                vocab_match[k] = v
    return vocab_match

def generate_vocab_match_v1_mutual(token_by_length):
    vocab_match = {}
    for _, tokens in token_by_length.items():
        tokens_copy = copy.deepcopy(tokens)
        
        if len(tokens_copy) % 2 == 0:
            mid_len = int(len(tokens_copy)/2)
            first_half = tokens_copy[:mid_len]
            second_half = tokens_copy[mid_len:]
            random.shuffle(first_half)
            random.shuffle(second_half)
            for i in range(len(first_half)):
                vocab_match[first_half[i]] = second_half[i]
                vocab_match[second_half[i]] = first_half[i]
        else:
            extra_token = tokens_copy[-1]
            tokens_copy_shorten = tokens_copy[:-1]
            mid_len = int(len(tokens_copy)/2)
            first_half = tokens_copy_shorten[:mid_len]
            second_half = tokens_copy_shorten[mid_len:]
            random.shuffle(first_half)
            random.shuffle(second_half)
            for i in range(len(first_half)):
                vocab_match[first_half[i]] = second_half[i]
                vocab_match[second_half[i]] = first_half[i]
            vocab_match[extra_token] = extra_token
    return vocab_match

def generate_vocab_match_v2_mutual(token_by_length, token_frequency_map, token_lemma_map):
    """
    TODO: fix bugs
    """
    vocab_match = {}
    for _, tokens in token_by_length.items():
        tokens_copy = copy.deepcopy(tokens)
        
        # token_frequency_map, token_lemma_map)
        
        token_freq_tu = []
        for t in tokens:
            token_freq_tu.append((t, token_frequency_map[t]))
        token_freq_tu = sorted(token_freq_tu, key=operator.itemgetter(1), reverse=True)
        
        matched_to = set([])
        for i in trange(0, len(token_freq_tu)):
            found = False
            for j in range(0, len(token_freq_tu)):
                word_i = token_freq_tu[i][0]
                word_j = token_freq_tu[j][0]
                if i != j and word_j not in matched_to and \
                    len(token_lemma_map[word_i].intersection(token_lemma_map[word_j])) == 0 and \
                    levenshteinDistance(word_i, word_j) > 0.3:
                    matched_to.add(word_j)
                    vocab_match[word_i] = word_j
                    found = True
                    break
            if not found:
                vocab_match[word_i] = word_i
            
    return vocab_match

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
            
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Bert."""

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt",
        "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt",
        "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt",
        "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt",
        "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt",
        "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}

class ModifiedBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = ModifiedBasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = ModifiedWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
        
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        token_dict = {}
        if self.do_basic_tokenize:

            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append([token])
                    token_dict[token] = [token]
                else:
                    split_tokens.append(self.wordpiece_tokenizer.tokenize(token))
                    token_dict[token] = self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
            
        return split_tokens, token_dict

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class ModifiedBasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class ModifiedWordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
# Load modules, mainly huggingface basic model handlers.
# Make sure you install huggingface and other packages properly.
from vocab_mismatch_utils import *
from data_formatter_utils import *
from datasets import DatasetDict
from datasets import Dataset
from datasets import load_dataset
from datasets import list_datasets
import transformers
import pandas as pd
import operator
from collections import OrderedDict
from tqdm import tqdm, trange
from scipy import stats

from collections import Counter
import json

from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef

import logging
logger = logging.getLogger(__name__)

import os
os.environ["TRANSFORMERS_CACHE"] = "../huggingface_cache/" # Not overload common dir 
                                                           # if run in shared resources.

import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from functools import partial

from vocab_mismatch_utils import *
basic_tokenizer = ModifiedBasicTokenizer()
from models.modeling_bert import CustomerizedBertForSequenceClassification

FILENAME_CONFIG = {
    "sst3" : "sst-tenary",
    "cola" : "cola",
    "mnli" : "mnli",
    "snli" : "snli",
    "mrpc" : "mrpc",
    "qnli" : "qnli",
    "conll2003" : "conll2003",
    "en_ewt" : "en_ewt"
}
TASK_CONFIG = {
    "wiki-text": ("text", None),
    "sst3": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "conll2003" : ("tokens", None),
    "en_ewt" : ("tokens", None)
}
# NUM_LABELS = 2 if args.task_name == "cola" or args.task_name == "mrpc" or args.task_name == "qnli" else 3
NUM_LABELS = {
    "sst3" : 3,
    "cola" : 2, 
    "mrpc" : 2,
    "qnli" : 2,
    "snli" : 3
}

SUBDIR_CONFIG = {
    "sst3" : "159274",
    "cola" : "8551", 
    "mrpc" : "3668",
    "qnli" : "104743",
    "snli" : "550152",
}

def generate_vocab_match_no_frequency_iv(token_by_length, token_frequency_map):
    vocab_match = {}
    tokens = list(task_token_frequency_map.keys())
    tokens_copy = copy.deepcopy(tokens)
    random.shuffle(tokens_copy)
    for i in range(len(tokens)):
        vocab_match[tokens[i]] = tokens_copy[i]
    return vocab_match

def generate_vocab_match_frequency_iv(token_by_length, token_frequency_map):
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
                    levenshteinDistance(word_i, word_j) > 0.3:
                    matched_to.add(word_j)
                    vocab_match[word_i] = word_j
                    found = True
                    break
            if not found:
                vocab_match[word_i] = word_i
            
    return vocab_match

def generate_vocab_match_abstract(token_frequency_map):
    
    # we need to first get the swapping dictionary.
    # this tokenizer helps you to get piece length for each token
    modified_tokenizer = ModifiedBertTokenizer(
        vocab_file="../data-files/bert_vocab.txt")
    modified_basic_tokenizer = ModifiedBasicTokenizer()
    vocab_match = {}
    vocab_list = []
    for k,v in token_frequency_map.items():
        vocab_list.append(k)
    random.shuffle(vocab_list)
    
    abstract_matches = []
    abstract_len = 4
    for i in range(0, abstract_len):
        az_list = []
        for j in range(ord('a'), ord('z')+1):
            az_list.append(chr(j))
        abstract_matches.append(az_list)
    from itertools import product
    abstract_matches = product(*abstract_matches)
    good_abstract = []
    for match in abstract_matches:
        abstract = "".join(match)
        if len(modified_tokenizer.tokenize(abstract)[0][0]) == abstract_len:
            good_abstract.append(abstract)
            if len(good_abstract) != 0 and len(good_abstract) % 10000 == 0:
                print(f"generating abstract token in progress: {len(good_abstract)}")
    
    assert len(good_abstract) >= len(vocab_list)
    
    for i in range(0, len(vocab_list)):
        vocab_match[vocab_list[i]] = good_abstract[i]
    return vocab_match

def generate_vocab_match_no_frequency_oov(wiki_token_frequency_map, 
                                          token_frequency_map,
                                          match_high=False, 
                                          match_similar=False):
    vocab_match = {}
    wiki_vocab_to_use = []
    if match_similar:
        in_vocab_rank = []
        for k, v in token_frequency_map.items():
            in_vocab_rank.append(k)
        wiki_tuples = []
        for k, v in wiki_token_frequency_map.items():
            if k not in task_token_frequency_map.keys():
                wiki_tuples.append((k, v))
        wiki_tuples = random.sample(wiki_tuples, k=len(in_vocab_rank))
        wiki_tuples = sorted(wiki_tuples, key=lambda x: (x[1],x[1]), reverse=True)
        for i in range(len(in_vocab_rank)):
            vocab_match[in_vocab_rank[i]] = wiki_tuples[i][0]
    else:
        if not match_high:
            for k, v in wiki_token_frequency_map.items():
                if k not in task_token_frequency_map:
                    if v == 1:
                        wiki_vocab_to_use.append(k)
            random.shuffle(wiki_vocab_to_use)
            freq_idx = 0
            for k, v in task_token_frequency_map.items():
                vocab_match[k] = wiki_vocab_to_use[freq_idx]
                freq_idx += 1
        else:
            in_vocab_rank = []
            for k, v in token_frequency_map.items():
                in_vocab_rank.append(k)
            in_vocab_rank = in_vocab_rank[::-1] # reverse the order
            idx = 0
            for k, v in wiki_token_frequency_map.items():
                if k not in task_token_frequency_map:
                    wiki_vocab_to_use.append(k)
                    idx += 1
                    if idx == len(in_vocab_rank):
                        break
            assert len(wiki_vocab_to_use)==len(in_vocab_rank)
            for i in range(len(wiki_vocab_to_use)):
                vocab_match[in_vocab_rank[i]] = wiki_vocab_to_use[i]
    return vocab_match

def random_corrupt(task, tokenizer, vocab_match, example):
    # for tasks that have single sentence
    if task == "sst3" or task == "wiki-text" or task == "cola":
        original_sentence = example[TASK_CONFIG[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][0]] = corrupted_sentence
    # for tasks that have two sentences
    elif task == "mrpc" or task == "mnli" or task == "snli" or task == "qnli":
        original_sentence = example[TASK_CONFIG[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][0]] = corrupted_sentence
        
        original_sentence = example[TASK_CONFIG[task][1]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[TASK_CONFIG[task][1]] = corrupted_sentence
    elif task == "conll2003" or task == "en_ewt":
        original_tokens = example[TASK_CONFIG[task][0]]
        corrupted_tokens = [vocab_match[t] for t in original_tokens]
        example[TASK_CONFIG[task][0]] = corrupted_tokens
    else:
        print(f"task={task} not supported yet!")
    return example

def plot_dist(vocab, map1, map2, facecolor='b', post_fix="mismatched"):
    freq_diff = []
    for k, v in vocab.items():
        diff = abs(map1[k] - map2[v])
        # print(diff)
        freq_diff.append(diff)
    fig = plt.figure(figsize=(8,3.5))
    ax = fig.add_subplot(111)
    g = ax.hist(freq_diff, bins=50, facecolor=facecolor, alpha=0.8)
    # plt.grid(True)
    # plt.grid(color='black', linestyle='-.')
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.set_yscale('log')
    plt.xlabel(f"Difference in Frequencies")
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ylabel("Frequency (LOG)")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%1.0e'))
    plt.tight_layout()
    plt.show()
    
def cosine_sim_distance(v1, v2):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim = cos(v1,v2)
    return sim

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def in_vocab_proximity(embeddings, vocab, return_dict=True, translate_dict=None):
    proximity = {}
    for (ex_index, example) in enumerate(tqdm(vocab.items())):
        word_index, word = example
        word_embedding = embeddings[word_index]
        repeat_word_embedding = [word_embedding*len(vocab)]
        repeat_word_embedding = torch.stack(repeat_word_embedding, dim=0)
        similarity_score = cosine_sim_distance(repeat_word_embedding, embeddings)
        # record the scores
        proximity[word] = []
        for i in range(len(vocab)):
             proximity[word] += [(vocab[i], similarity_score[i].tolist())]
    if return_dict:
        proximity_formatted = {}
        for k, v in proximity.items():
            if translate_dict is not None:
                proximity_formatted[translate_dict[k]] = {}
                for p in v:
                    if p[0] != k:
                        proximity_formatted[translate_dict[k]][translate_dict[p[0]]] = p[1]
            else:
                proximity_formatted[k] = {}
                for p in v:
                    if p[0] != k:
                        proximity_formatted[k][p[0]] = p[1]
        return proximity_formatted
    return proximity

def proximity_correlation(bert_proximity, testing_proximity):
    pair_corr = {}
    test_vocab = set(testing_proximity.keys())
    for (ex_index, word_lookup) in enumerate(tqdm(bert_proximity.keys())):
        distance_pairs = []
        for k, v in bert_proximity[word_lookup].items():
            if word_lookup in test_vocab:
                word_testing_proximity = testing_proximity[word_lookup]
                if k in word_testing_proximity:
                    testing_v = word_testing_proximity[k]
                    distance_p = (v, testing_v)
                    distance_pairs.append(distance_p)
        x = [p[0] for p in distance_pairs]
        y = [p[1] for p in distance_pairs]
        if len(x) != 0 and len(y) != 0:
            # I think we cannot use correlation here.
            # We need to use absolute distance.
            # mse = 0.0
            # for i in range(len(x)):
            #     mse += (x[i]-y[i])*(x[i]-y[i])
            # mse /= len(x)
            pair_corr[word_lookup] = stats.pearsonr(x, y)[0]
            # pair_corr[word_lookup] = mse
    dist_to_plot = []
    for k, v in pair_corr.items():
        dist_to_plot += [v]
    return dist_to_plot
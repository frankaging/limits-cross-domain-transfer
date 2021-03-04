import csv
import json
import glob
from nltk.tree import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
import pandas as pd
import random


NEG = "0"
POS = "1"
NEUTRAL = "2"


def write_tsv(*datasets, output_filename, fieldnames=['text', 'label', 'source']):
    all_data = []
    for dataset in datasets:
        all_data += dataset
    random.shuffle(all_data)
    with open(output_filename, "wt") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

external_dataset_dir = os.path.join("..", "..", "external-datasets")

yelp_reviews_dirname = os.path.join(external_dataset_dir, "yelp_review_full_csv")

amazon_reviews_dirname = os.path.join(external_dataset_dir, "amazon_review_full_csv")

sst_dirname = os.path.join(external_dataset_dir, "trees")

cr_dirname = os.path.join(external_dataset_dir, "custrev")

imdb_dirname = os.path.join(external_dataset_dir, "aclImdb")

our_dataset_home = os.path.join(external_dataset_dir, "dynasent-v1")

round0_output_dirname = os.path.join(our_dataset_home, "round0")

round1_output_dirname = os.path.join(our_dataset_home, "round1")

round2_output_dirname = os.path.join(our_dataset_home, "round2")

external_output_dirname = os.path.join("..", "data-files")


## Readers


### Large review corpora


ternary_map = {1: NEG, 2: NEG, 3: NEUTRAL, 4: POS, 5:  POS}


three_only_map = {3: NEUTRAL}


def process_large_review(
        src_filename,
        output_filename=None,
        label_map=ternary_map,
        sample_size=None,
        line_nums=None):
    if line_nums is not None:
        line_nums = set(range(*line_nums))
    source = src_filename.replace(other_datasets_dirname, "").strip("/")
    data = []
    with open(src_filename) as f:
        for line_num, row in enumerate(csv.reader(f)):
            if line_nums is None or line_num in line_nums:
                rating = row[0]
                text = row[-1]
                label = label_map.get(int(rating))
                if label and text.strip():
                    data.append({"text": text, "label": label, "source": source})
    if sample_size is not None:
        random.shuffle(data)
        data = data[: sample_size]
    return data


### SST


def only_three(y):
     return NEUTRAL if y == "2" else None


def full_ternary_class_func(y):
    if y in {"0", "1"}:
        return NEG
    elif y in {"3", "4"}:
        return POS
    elif y == "2":
        return NEUTRAL
    else:
        return None


DETOKENIZER = TreebankWordDetokenizer()


def sst_reader(src_filename, class_func=None, include_subtrees=True):
    if class_func is None:
        class_func = lambda x: x
    with open(src_filename) as f:
        for line in f:
            tree = Tree.fromstring(line)
            if include_subtrees:
                for subtree in tree.subtrees():
                    label = class_func(subtree.label())
                    yield (_sst_detokenize(subtree), label)
            else:
                label = class_func(tree.label())
                yield (_sst_detokenize(tree), label)

def _sst_detokenize(tree):
    return DETOKENIZER.detokenize(tree.leaves())


def process_sst(src_filename, class_func, include_subtrees=True, dedup=True):
    data = []
    for text, label in sst_reader(src_filename, class_func=class_func, include_subtrees=include_subtrees):
        data.append({'text': text, "label": label, "source": "sst"})
    if dedup:
        data = {d['text']: d for d in data}
        data = list(data.values())
    return data


### Customer reviews


def process_cr(src_dirname):
    data = []
    data += [(l, POS) for l in open(os.path.join(src_dirname, 'custrev.pos')).read().splitlines()]
    data += [(l, NEG) for l in open(os.path.join(src_dirname, 'custrev.neg')).read().splitlines()]
    data = [{'text': text, 'label': label, 'source': 'custrev'} for text, label in data if text.strip()]
    return data


### IMDB


def process_imdb(imdb_dirname):
    data = []
    for dirname, label in [('pos', POS), ('neg', NEG)]:
        for filename in glob.glob(os.path.join(imdb_dirname, dirname, "*.txt")):
            with open(filename) as f:
                text = f.read()
                if text.strip():
                    data.append({"text": text, "label": label, "source": "imdb"})
    return data


### Our data


def load_dynasent(src_filename, dist_labels=False):
    data = []
    # get the filename of basename without the ext postfix
    source = os.path.splitext(os.path.basename(src_filename))[0]
    label_map = {"negative": "0", "positive": "1", "neutral": "2"}
    with open(src_filename) as f:
        for line in f:
            d = json.loads(line)
            text = d['sentence']
            if dist_labels:
                for label, workers in d['label_distribution'].items():
                    labels = ["positive", "negative"] if label == "mixed" else [label]
                    for x in labels:
                        x = label_map[x]
                        for w in workers:
                            data.append({"text": text, "label": x, "source": source})
            else:
                label = d['gold_label']
                label = label_map.get(label)
                if label is not None:
                    data.append({"text": text, "label": label, "source": source})
    return data
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load modules, mainly huggingface basic model handlers.
# Make sure you install huggingface and other packages properly.
from collections import Counter
import json

from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import logging
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import argparse
import numpy as np
from datasets import load_dataset, load_metric
from datasets import Dataset
import pandas as pd

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
)
from transformers.trainer_utils import is_main_process


# In[ ]:


class HuggingFaceRoBERTaBase:
    """
    An extension for evaluation based off the huggingface module.
    """
    def __init__(self, tokenizer, model, task_config):
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.model = model
        
    def evaluation(self, data_path, training_args, max_length=128, csv_source="all"):

        print("*** Evaluate with %s ***"%(data_path))
        
        eval_df = pd.read_csv(data_path , delimiter="\t")
        # if source is not all, we need to filter rows based on the source column
        if csv_source != "all":
            eval_df = eval_df[eval_df["source"]==csv_source]
        datasets = {}
        datasets["validation"] = Dataset.from_pandas(eval_df)
        
        label_list = datasets["validation"].unique("label")
        label_list.sort()  # Let's sort it for determinism

        padding = "max_length"
        sentence1_key, sentence2_key = self.task_config
        label_to_id = None
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[l] for l in examples["label"]]
            return result
        datasets["validation"] = datasets["validation"].map(preprocess_function, batched=True)
        
        eval_dataset = datasets["validation"]
        
        metric = load_metric("glue", "sst2") # any glue task will do the job, just for eval loss
        
        def dynasent_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result_to_print = classification_report(p.label_ids, preds, digits=5, output_dict=True)
            print(classification_report(p.label_ids, preds, digits=5))
            result_to_return = metric.compute(predictions=preds, references=p.label_ids)
            result_to_return["Macro-F1"] = result_to_print["macro avg"]["f1-score"]
            return result_to_return
        
        # Initialize our Trainer. We are only intersted in evaluations
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=dynasent_compute_metrics,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator
        )
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        
        print("*** Loss and GLUE AUC ***")
        print(eval_result)


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default="dynasent",
                        type=str)
    parser.add_argument("--data_path",
                        default="../datasets/round0/round0-dev.tsv",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_path",
                        default="../saved_models/pytorch_model.bin",
                        type=str,
                        help="The pretrained model binary file.")
    parser.add_argument("--model_type",
                        default="roberta-base",
                        type=str,
                        help="The pretrained model binary file.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--cache_dir",
                        default="../tmp/",
                        type=str,
                        help="Cache directory for the evaluation pipeline (not HF cache).")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    parser.add_argument("--per_device_eval_batch_size",
                        default=8,
                        type=int,
                        help="The batch size per device for evaluation.")
    parser.add_argument("--is_tensorboard",
                        default=False,
                        action='store_true',
                        help="If tensorboard is connected.")
    parser.add_argument("--csv_source",
                        default="all",
                        type=str,
                        help="If the csv file has a source column, only source from this indicated source"
                             " will be used to evaluate.")
    parser.add_argument("--embeddings_path",
                        default="",
                        type=str,
                        help="The embedding file to swap.")
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    # os.environ["WANDB_DISABLED"] = "false" if args.is_tensorboard else "true"
    os.environ["TRANSFORMERS_CACHE"] = "../huggingface_cache/"
    # if cache does not exist, create one
    if not os.path.exists(os.environ["TRANSFORMERS_CACHE"]): 
        os.makedirs(os.environ["TRANSFORMERS_CACHE"])

    training_args = TrainingArguments("tmp_trainer")
    training_args.no_cuda = args.no_cuda
    training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    training_args.per_gpu_eval_batch_size = args.per_device_eval_batch_size
    training_args_dict = training_args.to_dict()
    _n_gpu = training_args_dict["_n_gpu"]
    del training_args_dict["_n_gpu"]
    training_args_dict["n_gpu"] = _n_gpu
    HfParser = HfArgumentParser((TrainingArguments))
    training_args = HfParser.parse_dict(training_args_dict)[0]

    TASK_CONFIG = {
        "classification": ("text", None)
    }

    # Load pretrained model and tokenizer
    NUM_LABELS = 3
    MAX_SEQ_LEN = 128
    config = AutoConfig.from_pretrained(
        args.model_type,
        num_labels=3,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        use_fast=False,
        cache_dir=args.cache_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir
    )
    if len(args.embeddings_path) != 0:
        logger.info("***** Loading an new embedding file to the model *****")
        logger.info("***** You are evaluating sort of zero-shot here!!! *****")
        transformed_weight = torch.load(args.embeddings_path)
        model.bert.embeddings.word_embeddings.weight.data = transformed_weight.data
    
    eval_pipeline = HuggingFaceRoBERTaBase(tokenizer, model, TASK_CONFIG[args.task_name])
        
    eval_pipeline.evaluation(args.data_path, training_args, max_length=args.max_seq_length, 
                             csv_source=args.csv_source)


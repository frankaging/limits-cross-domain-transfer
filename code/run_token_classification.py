#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load modules, mainly huggingface basic model handlers.
# Make sure you install huggingface and other packages properly.
from collections import Counter
import json

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
from typing import Optional
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import DatasetDict
from datasets import ClassLabel

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast
)
from transformers.trainer_utils import is_main_process, EvaluationStrategy
from functools import partial

from vocab_mismatch_utils import *
basic_tokenizer = ModifiedBasicTokenizer()
from models.modeling_bert import CustomerizedBertForSequenceClassification


# In[ ]:


def generate_training_args(args, inoculation_step):
    training_args = TrainingArguments("tmp_trainer")
    training_args.no_cuda = args.no_cuda
    training_args.seed = args.seed
    training_args.do_train = args.do_train
    training_args.do_eval = args.do_eval
    training_args.output_dir = os.path.join(args.output_dir, str(inoculation_step)+"-sample")
    training_args.evaluation_strategy = args.evaluation_strategy # evaluation is done after each epoch
    training_args.metric_for_best_model = args.metric_for_best_model
    training_args.greater_is_better = args.greater_is_better
    training_args.logging_dir = args.logging_dir
    training_args.task_name = args.task_name
    training_args.learning_rate = args.learning_rate
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    training_args.num_train_epochs = args.num_train_epochs # this is the maximum num_train_epochs, we set this to be 100.
    training_args.eval_steps = args.eval_steps
    training_args.logging_steps = args.logging_steps
    training_args.load_best_model_at_end = args.load_best_model_at_end
    if args.save_total_limit != -1:
        # only set if it is specified
        training_args.save_total_limit = args.save_total_limit
    import datetime
    date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)
    run_name = "{0}_{1}_{2}_{3}_mlen_{4}_lr_{5}_seed_{6}_metrics_{7}".format(
        args.run_name,
        args.task_name,
        args.model_type,
        date_time,
        args.max_seq_length,
        args.learning_rate,
        args.seed,
        args.metric_for_best_model
    )
    training_args.run_name = run_name
    training_args_dict = training_args.to_dict()
    # for PR
    _n_gpu = training_args_dict["_n_gpu"]
    del training_args_dict["_n_gpu"]
    training_args_dict["n_gpu"] = _n_gpu
    HfParser = HfArgumentParser((TrainingArguments))
    training_args = HfParser.parse_dict(training_args_dict)[0]

    if args.model_path == "":
        args.model_path = args.model_type
        if args.model_type == "":
            assert False # you have to provide one of them.
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    return training_args


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--wandb_proj_name",
                        default="",
                        type=str)
    parser.add_argument("--task_name",
                        default="conll2003",
                        type=str)
    parser.add_argument("--token_type",
                        default="pos",
                        type=str)
    parser.add_argument("--run_name",
                        default="inoculation-run",
                        type=str)
    parser.add_argument("--inoculation_data_path",
                        default="../data-files/sst-tenary/sst-tenary-train.tsv",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_path",
                        default="../data-files/sst-tenary/sst-tenary-dev.tsv",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_path",
                        default="",
                        type=str,
                        help="The pretrained model binary file.")
    parser.add_argument("--model_type",
                        default="bert-base-uncased",
                        type=str,
                        help="The pretrained model binary file.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--evaluation_strategy",
                        default="steps",
                        type=str,
                        help="When you evaluate your training model on eval set.")
    parser.add_argument("--cache_dir",
                        default="../tmp/",
                        type=str,
                        help="Cache directory for the evaluation pipeline (not HF cache).")
    parser.add_argument("--logging_dir",
                        default="../tmp/",
                        type=str,
                        help="Logging directory.")
    parser.add_argument("--output_dir",
                        default="../results/",
                        type=str,
                        help="Output directory of this training process.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--metric_for_best_model",
                        default="f1",
                        type=str,
                        help="The metric to use to compare two different models.")
    parser.add_argument("--greater_is_better",
                        default=True,
                        action='store_true',
                        help="Whether the `metric_for_best_model` should be maximized or not.")
    parser.add_argument("--is_tensorboard",
                        default=False,
                        action='store_true',
                        help="If tensorboard is connected.")
    parser.add_argument("--load_best_model_at_end",
                        default=False,
                        action='store_true',
                        help="Whether load best model and evaluate at the end.")
    parser.add_argument("--label_all_tokens",
                        default=False,
                        action='store_true',
                        help="Whether to put the label for one word on all tokens of generated by that word or just on the "
                             "one (in which case the other tokens will have a padding index).")
    parser.add_argument("--eval_steps",
                        default=10,
                        type=float,
                        help="The total steps to flush logs to wandb specifically.")
    parser.add_argument("--logging_steps",
                        default=10,
                        type=float,
                        help="The total steps to flush logs to wandb specifically.")
    parser.add_argument("--save_total_limit",
                        default=-1,
                        type=int,
                        help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output dir.")
    # these are arguments for inoculations
    parser.add_argument("--inoculation_patience_count",
                        default=5,
                        type=int,
                        help="If the evaluation metrics is not increasing with maximum this step number, the training will be stopped.")
    parser.add_argument("--inoculation_step_sample_size",
                        default=0.05,
                        type=float,
                        help="For each step, how many more adverserial samples you want to add in.")
    parser.add_argument("--per_device_train_batch_size",
                        default=8,
                        type=int,
                        help="")
    parser.add_argument("--per_device_eval_batch_size",
                        default=8,
                        type=int,
                        help="")
    parser.add_argument("--preprocessing_num_workers",
                        default=8,
                        type=int,
                        help="")
    parser.add_argument("--eval_sample_limit",
                        default=-1,
                        type=int,
                        help="")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="The total number of epochs for training.")
    parser.add_argument("--no_pretrain",
                        default=False,
                        action='store_true',
                        help="Whether to use pretrained model if provided.")
    parser.add_argument("--train_embeddings_only",
                        default=False,
                        action='store_true',
                        help="If only train embeddings not the whole model.")
    parser.add_argument("--train_linear_layer_only",
                        default=False,
                        action='store_true',
                        help="If only train embeddings not the whole model.")
    # these are arguments for scrambling texts
    parser.add_argument("--scramble_proportion",
                        default=0.0,
                        type=float,
                        help="What is the percentage of text you want to scramble.")
    parser.add_argument("--eval_with_scramble",
                        default=False,
                        action='store_true',
                        help="If you are also evaluating with scrambled texts.")
    parser.add_argument("--overwrite_cache",
                        default=False,
                        action='store_true',
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--n_layer_to_finetune",
                        default=-1,
                        type=int,
                        help="Indicate a number that is less than original layer if you only want to finetune with earlier layers only.")
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    # os.environ["WANDB_DISABLED"] = "NO" if args.is_tensorboard else "YES" # BUG
    os.environ["TRANSFORMERS_CACHE"] = "../huggingface_inoculation_cache/"
    os.environ["WANDB_PROJECT"] = f"{args.task_name}_bertonomy"
    # if cache does not exist, create one
    if not os.path.exists(os.environ["TRANSFORMERS_CACHE"]): 
        os.makedirs(os.environ["TRANSFORMERS_CACHE"])
    TASK_CONFIG = {
        "conll2003" : ("tokens", None),
        "en_ewt" : ("tokens", None)
    }
    # we need some prehandling here for token classifications!
    # we use panda loader now, to make sure it is backward compatible
    # with our file writer.
    pd_format = True
    if "data-files" not in args.inoculation_data_path:
        # here, we download from the huggingface server probably!
        if args.inoculation_data_path == "en_ewt":
            dataset = load_dataset("universal_dependencies", args.inoculation_data_path, cache_dir=args.cache_dir) # make sure cache dir is self-contained!
        else:
            dataset = load_dataset(args.inoculation_data_path, cache_dir=args.cache_dir) # make sure cache dir is self-contained!
        inoculation_train_df = dataset["train"]
        eval_df = dataset["validation"]
    else:  
        pd_format = True
        if args.inoculation_data_path.split(".")[-1] != "tsv":
            if len(args.inoculation_data_path.split(".")) > 1:
                logger.info(f"***** Loading pre-loaded datasets from the disk directly! *****")
                pd_format = False
                datasets = DatasetDict.load_from_disk(args.inoculation_data_path)
                inoculation_step_sample_size = int(len(datasets["train"]) * args.inoculation_step_sample_size)
                logger.info(f"***** Inoculation Sample Count: %s *****"%(inoculation_step_sample_size))
                # this may not always start for zero inoculation
                training_args = generate_training_args(args, inoculation_step=inoculation_step_sample_size)
                datasets["train"] = datasets["train"].shuffle(seed=args.seed)
                inoculation_train_df = datasets["train"].select(range(inoculation_step_sample_size))
                eval_df = datasets["validation"]
                datasets["validation"] = datasets["validation"].shuffle(seed=args.seed)
                if args.eval_sample_limit != -1:
                    datasets["validation"] = datasets["validation"].select(range(args.eval_sample_limit))
            else:
                logger.info(f"***** Loading downloaded huggingface datasets: {args.inoculation_data_path}! *****")
                pd_format = False
                if args.inoculation_data_path in ["sst3", "cola", "mnli", "snli", "mrps", "qnli", "conll2003", "en_ewt"]:
                    pass
                raise NotImplementedError()
    
    text_column_name = TASK_CONFIG[args.task_name][0]
    label_column_name = args.token_type
    features = inoculation_train_df.features
    
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(inoculation_train_df[label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    MAX_SEQ_LEN = args.max_seq_length
    training_args = generate_training_args(args, inoculation_step=0)
    config = AutoConfig.from_pretrained(
        args.model_type,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir
    )
    if args.n_layer_to_finetune != -1:
        # then we are only finetuning n-th layer, not all the layers
        if args.n_layer_to_finetune > config.num_hidden_layers:
            logger.info(f"***** WARNING: You are trying to train with first {args.n_layer_to_finetune} layers only *****")
            logger.info(f"***** WARNING: But the model has only {config.num_hidden_layers} layers *****")
            logger.info(f"***** WARNING: Training with all layers instead! *****")
            pass # just to let it happen, just train it with all layers
        else:
            # overwrite
            logger.info(f"***** WARNING: You are trying to train with first {args.n_layer_to_finetune} layers only *****")
            logger.info(f"***** WARNING: But the model has only {config.num_hidden_layers} layers *****")
            config.num_hidden_layers = args.n_layer_to_finetune
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        use_fast=True,
        cache_dir=args.cache_dir
    )
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )
    
    if args.no_pretrain:
        logger.info("***** Training new model from scratch *****")
        model = AutoModelForTokenClassification.from_config(config)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_path,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir
        )
        
    if args.train_embeddings_only:
        logger.info("***** We only train embeddings, not other layers *****")
        for name, param in model.named_parameters():
            if 'word_embeddings' not in name: # only word embeddings
                param.requires_grad = False
    
    if args.train_linear_layer_only:
        logger.info("***** We only train classifier head, not other layers *****")
        for name, param in model.named_parameters():
            if 'classifier' not in name: # only word embeddings
                param.requires_grad = False
        
    logger.info(f"***** TASK NAME: {args.task_name} *****")
    datasets = {}
    datasets["train"] = inoculation_train_df
    datasets["validation"] = eval_df
    logger.info(f"***** Train Sample Count (Verify): %s *****"%(len(datasets["train"])))
    logger.info(f"***** Valid Sample Count (Verify): %s *****"%(len(datasets["validation"])))
    
    padding = "max_length"
    
    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # preparing datasets
    datasets["train"] = datasets["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
    )
    
    datasets["validation"] = datasets["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=padding)
    
    # Metrics
    ner_metric = load_metric("seqeval")

    def compute_metrics_ner(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = ner_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    def compute_metrics_pos(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_predictions = [item for sublist in true_predictions for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]
        
        result_to_print = classification_report(true_labels, true_predictions, digits=5, output_dict=True)
        print(classification_report(true_labels, true_predictions, digits=5))
        
        return {
            "f1": result_to_print["macro avg"]["f1-score"],
            "accuracy": result_to_print["accuracy"],
        }
        
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_pos if args.task_name == "en_ewt" else compute_metrics_ner,
    )
    
    # Early stop
    if args.inoculation_patience_count != -1:
        trainer.add_callback(EarlyStoppingCallback(args.inoculation_patience_count))

    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics["train_samples"] = len(train_dataset)

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)


# In[ ]:





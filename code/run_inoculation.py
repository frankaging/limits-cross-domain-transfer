#!/usr/bin/env python
# coding: utf-8

# #### Pipeline for analyzing adverserial examples via fine-tuning

# In[1]:


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


# In[2]:


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

class HuggingFaceRoBERTaBase:
    """
    An extension for evaluation based off the huggingface module.
    """
    def __init__(self, tokenizer, model, task_name, task_config):
        self.task_name = task_name
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.model = model
        
    def train(self, inoculation_train_df, eval_df, model_path, training_args, max_length=128,
              inoculation_patience_count=5, pd_format=True, 
              scramble_proportion=0.0, eval_with_scramble=False):

        if pd_format:
            datasets = {}
            datasets["train"] = Dataset.from_pandas(inoculation_train_df)
            datasets["validation"] = Dataset.from_pandas(eval_df)
        else:
            datasets = {}
            datasets["train"] = inoculation_train_df
            datasets["validation"] = eval_df
        logger.info(f"***** Train Sample Count (Verify): %s *****"%(len(datasets["train"])))
        logger.info(f"***** Valid Sample Count (Verify): %s *****"%(len(datasets["validation"])))
    
        label_list = datasets["validation"].unique("label")
        label_list.sort()  # Let's sort it for determinism

        # we will scramble out input sentence here
        # TODO: we scramble both train and eval sets
        def scramble_inputs(proportion, example):
            original_text = example['text']
            original_sentence = basic_tokenizer.tokenize(original_text)
            max_length = len(original_sentence)
            scramble_length = int(max_length*proportion)
            scramble_start = random.randint(0, len(original_sentence)-scramble_length)
            scramble_end = scramble_start + scramble_length
            scramble_sentence = original_sentence[scramble_start:scramble_end]
            random.shuffle(scramble_sentence)
            scramble_text = original_sentence[:scramble_start] + scramble_sentence + original_sentence[scramble_end:]

            out_string = " ".join(scramble_text).replace(" ##", "").strip()
            example['text'] = out_string
            return example
        
        if scramble_proportion > 0.0:
            logger.info(f"You are scrambling the inputs to test syntactic feature importance!")
            datasets["train"] = datasets["train"].map(partial(scramble_inputs, scramble_proportion), 
                                                      batched=True)
            if eval_with_scramble:
                logger.info(f"You are scrambling the evaluation data as well!")
                datasets["validation"] = datasets["validation"].map(partial(scramble_inputs, scramble_proportion), 
                                                                    batched=True)
        
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
        datasets["train"] = datasets["train"].map(preprocess_function, batched=True)
        datasets["validation"] = datasets["validation"].map(preprocess_function, batched=True)
        
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]
        
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            
        metric = load_metric("glue", "sst2") # any glue task will do the job, just for eval loss
        
        def asenti_compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result_to_print = classification_report(p.label_ids, preds, digits=5, output_dict=True)
            print(classification_report(p.label_ids, preds, digits=5))
            mcc_scores = matthews_corrcoef(p.label_ids, preds)
            logger.info(f"MCC scores: {mcc_scores}.")
            result_to_return = metric.compute(predictions=preds, references=p.label_ids)
            result_to_return["Macro-F1"] = result_to_print["macro avg"]["f1-score"]
            result_to_return["MCC"] = mcc_scores
            return result_to_return

        # Initialize our Trainer. We are only intersted in evaluations
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=asenti_compute_metrics,
            tokenizer=self.tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator
        )
        # Early stop
        if inoculation_patience_count != -1:
            trainer.add_callback(EarlyStoppingCallback(inoculation_patience_count))
        
        # Training
        if training_args.do_train:
            logger.info("*** Training our model ***")
            trainer.train(
                model_path=model_path
            )
            trainer.save_model()  # Saves the tokenizer too for easy upload
        
        # Evaluation
        eval_results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            tasks = [self.task_name]
            eval_datasets = [eval_dataset]
            for eval_dataset, task in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in eval_result.items():
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")
                eval_results.update(eval_result)


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--wandb_proj_name",
                        default="",
                        type=str)
    parser.add_argument("--task_name",
                        default="sst3",
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
                        default="Macro-F1",
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
        "sst3": ("text", None),
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "snli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence")
    }
    # Load pretrained model and tokenizer
    NUM_LABELS = 2 if args.task_name == "cola" or args.task_name == "mrpc" or args.task_name == "qnli" else 3
    MAX_SEQ_LEN = args.max_seq_length
    training_args = generate_training_args(args, inoculation_step=0)
    config = AutoConfig.from_pretrained(
        args.model_type,
        num_labels=NUM_LABELS,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        use_fast=False,
        cache_dir=args.cache_dir
    )
    if args.no_pretrain:
        logger.info("***** Training new model from scratch *****")
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
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
        
    train_pipeline = HuggingFaceRoBERTaBase(tokenizer, 
                                            model, args.task_name, 
                                            TASK_CONFIG[args.task_name])
    logger.info(f"***** TASK NAME: {args.task_name} *****")
    # we use panda loader now, to make sure it is backward compatible
    # with our file writer.
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
            if args.inoculation_data_path in ["sst3", "cola", "mnli", "snli", "mrps", "qnli"]:
                pass
            raise NotImplementedError()
    else:
        train_df = pd.read_csv(args.inoculation_data_path, delimiter="\t")
        eval_df = pd.read_csv(args.eval_data_path, delimiter="\t")
        inoculation_step_sample_size = int(len(train_df) * args.inoculation_step_sample_size)
        logger.info(f"***** Inoculation Sample Count: %s *****"%(inoculation_step_sample_size))
        # this may not always start for zero inoculation
        training_args = generate_training_args(args, inoculation_step=inoculation_step_sample_size)
        inoculation_train_df = train_df.sample(n=inoculation_step_sample_size, 
                                               replace=False, 
                                               random_state=args.seed) # seed here could not a little annoying.


    train_pipeline.train(inoculation_train_df, eval_df, 
                         args.model_path,
                         training_args, max_length=args.max_seq_length,
                         inoculation_patience_count=args.inoculation_patience_count, pd_format=pd_format, 
                         scramble_proportion=args.scramble_proportion, eval_with_scramble=args.eval_with_scramble)


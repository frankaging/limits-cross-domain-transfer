#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import csv

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from vocab_mismatch_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_metrics = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mnli_mismatched": ["accuracy"],
    "mnli_matched": ["accuracy"],
    "mnli-mm": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr"],
    "wnli": ["accuracy"],
    "hans": ["accuracy"],
}

logger = logging.getLogger(__name__)
from functools import partial

def random_corrupt(task, tokenizer, vocab_match, example):
    # for tasks that have single sentence
    if task == "sst3" or task == "wiki-text" or task == "cola" or task == "sst2":
        original_sentence = example[task_to_keys[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[task_to_keys[task][0]] = corrupted_sentence
    # for tasks that have two sentences
    elif task == "mrpc" or task == "mnli" or task == "snli" or task == "qnli" or task == "qqp" or task == "rte" or task == "stsb" or task == "wnli":
        original_sentence = example[task_to_keys[task][0]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[task_to_keys[task][0]] = corrupted_sentence
        
        original_sentence = example[task_to_keys[task][1]]
        if original_sentence != None and original_sentence.strip() != "" and original_sentence.strip() != "None":
            corrupted_sentence = corrupt_translator(original_sentence, tokenizer, vocab_match)
            example[task_to_keys[task][1]] = corrupted_sentence
    elif task == "conll2003" or task == "en_ewt":
        original_tokens = example[task_to_keys[task][0]]
        corrupted_tokens = [vocab_match[t] for t in original_tokens]
        example[task_to_keys[task][0]] = corrupted_tokens
    else:
        print(f"task={task} not supported yet!")
        assert False
    return example


# In[ ]:


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    # customized params
    reverse_order: bool = field(
        default=False, metadata={"help": "TODO"}
    )
    random_order: bool = field(
        default=False, metadata={"help": "TODO"}
    )  
    token_swapping: bool = field(
        default=False, metadata={"help": "TODO"}
    ) 
    word_swapping: bool = field(
        default=False, metadata={"help": "TODO"}
    ) 
    word_freq_swapping: bool = field(
        default=False, metadata={"help": "TODO"}
    )
    reinit_avg_embeddings: bool = field(
        default=False, metadata={"help": "TODO"}
    ) 
    reinit_embeddings: bool = field(
        default=False, metadata={"help": "TODO"}
    )  
    swap_vocab_file: str = field(
        default="../data-files/wikitext-15M-vocab.json",
        metadata={"help": "TODO"},
    )
    swap_token_length_file: str = field(
        default="../data-files/wikitext-15M-token-length.json",
        metadata={"help": "TODO"},
    )
    pre_swap_map_file: str = field(
        default="../data-files/wikitext-15M-pre-swap-map.json",
        metadata={"help": "TODO"},
    )
    inoculation_p: float = field(
        default=1.0, metadata={"help": "TODO"}
    )  
        
#     def __post_init__(self):
#         if self.task_name is not None:
#             self.task_name = self.task_name.lower()
#             if self.task_name not in task_to_keys.keys():
#                 raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
#         elif self.dataset_name is not None:
#             pass
#         elif self.train_file is None or self.validation_file is None:
#             raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
#         else:
#             train_extension = self.train_file.split(".")[-1]
#             assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
#             validation_extension = self.validation_file.split(".")[-1]
#             assert (
#                 validation_extension == train_extension
#             ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


# In[ ]:


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # we hard-code these arguments just to make life eaiser!
    os.environ["TRANSFORMERS_CACHE"] = "./huggingface_inoculation_cache/"
    # with this, we could log into different places and interprete results directly.
    if training_args.do_train:
        os.environ["WANDB_PROJECT"] = f"ICLR_GLUE_train"
    elif training_args.do_eval:
        os.environ["WANDB_PROJECT"] = f"ICLR_GLUE_eval"
    elif training_args.do_predict:
        os.environ["WANDB_PROJECT"] = f"ICLR_GLUE_predict"
    model_args.cache_dir = "./huggingface_inoculation_cache/"
    training_args.save_total_limit = 1
    training_args.output_dir = "../"
    data_args.max_seq_length = 128
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    if training_args.do_eval:
        # we need to automatically pick out the task name.
        task_name_list = model_args.model_name_or_path.split("_")
        for i in range(len(task_name_list)):
            if task_name_list[i] == "task":
                data_args.task_name = task_name_list[i+1]
                break
        assert data_args.task_name in list(task_to_keys.keys())
        out_glue_task = data_args.task_name
        out_model = "bert-base-uncased"
        if "albert-base-v2" in model_args.model_name_or_path:
            out_tokenizer_name = "albert-base-v2"
            out_reinit_embedding = True
        elif "bert-base-cased" in model_args.model_name_or_path:
            out_tokenizer_name = "bert-base-cased"
            out_reinit_embedding = True
        elif "flaubert_base_cased" in model_args.model_name_or_path:
            out_tokenizer_name = "flaubert/flaubert_base_cased"
            out_reinit_embedding = True
        elif "bert-base-dutch-cased" in model_args.model_name_or_path:
            out_tokenizer_name = "GroNLP/bert-base-dutch-cased"
            out_reinit_embedding = True
            
        elif "bert-base-uncased" in model_args.model_name_or_path:
            out_tokenizer_name = "bert-base-uncased"
            if "reinit_emb_True" in model_args.model_name_or_path:
                out_reinit_embedding = True
            else:
                out_reinit_embedding = False
        elif "deberta-base" in model_args.model_name_or_path:
            out_tokenizer_name = "microsoft/deberta-base"
            if "reinit_emb_True" in model_args.model_name_or_path:
                out_reinit_embedding = True
            else:
                out_reinit_embedding = False
                
        if "inoculation_1.0" in model_args.model_name_or_path:
            out_midtuning = True
        elif "inoculation_0.0" in model_args.model_name_or_path:
            out_midtuning = False
        if "data_wikitext-15M-en~fr@N~fr@V" in model_args.model_name_or_path:
            out_galactic_shift = "en~fr@N~fr@V"
        elif "data_wikitext-15M-en~jaktc@N~jaktc@V" in model_args.model_name_or_path:
            out_galactic_shift = "en~jaktc@N~jaktc@V"
        elif "data_wikitext-15M-en~fr@N~jaktc@V" in model_args.model_name_or_path:
            out_galactic_shift = "en~fr@N~jaktc@V"
        else:
            assert "data_wikitext-15M_" in model_args.model_name_or_path
            out_galactic_shift = "null"

        if "reverse_True" in model_args.model_name_or_path:
            out_reverse = True
        else:
            out_reverse = False
        if "random_True" in model_args.model_name_or_path:
            out_random = True
        else:
            out_random = False
        if "token_s_True" in model_args.model_name_or_path:
            out_token_s = True
        else:
            out_token_s = False
        if "word_s_True" in model_args.model_name_or_path:
            out_word_s = True
        else:
            out_word_s = False
        if "word_freq_s_True" in model_args.model_name_or_path:
            out_word_freq_s = True
        else:
            out_word_freq_s = False
        if "lr_4e-05" in model_args.model_name_or_path:
            out_lr = 4e-05
        elif "lr_2e-05" in model_args.model_name_or_path:
            out_lr = 2e-05
        else:
            out_lr = 2e-05
        
    # overwrite the training epoch a little.
    if data_args.task_name == "mrpc" or data_args.task_name == "wnli" or data_args.task_name == "cola" or data_args.task_name == "rte":
        training_args.num_train_epochs = 15
    else:
        training_args.num_train_epochs = 3
    # overwrite the learning rate.
    if "lr_8e-05" in model_args.model_name_or_path:
        training_args.learning_rate = 8e-05
    elif "lr_6e-05" in model_args.model_name_or_path:
        training_args.learning_rate = 6e-05
    elif "lr_4e-05" in model_args.model_name_or_path:
        training_args.learning_rate = 4e-05
    else:
        training_args.learning_rate = 2e-05
    
    # this is for evaluation!
    condition_name = model_args.model_name_or_path
    
    # overwrite tokenizer based on model name.
    if "albert-base-v2" in model_args.model_name_or_path:
        model_args.tokenizer_name = "albert-base-v2"
    elif "bert-base-cased" in model_args.model_name_or_path:
        model_args.tokenizer_name = "bert-base-cased"
    elif "flaubert_base_cased" in model_args.model_name_or_path:
        model_args.tokenizer_name = "flaubert/flaubert_base_cased"
    elif "bert-base-dutch-cased" in model_args.model_name_or_path:
        model_args.tokenizer_name = "GroNLP/bert-base-dutch-cased"
    elif "deberta-base" in model_args.model_name_or_path:
        model_args.tokenizer_name =  = "microsoft/deberta-base"
    else:
        model_args.tokenizer_name = model_args.model_name_or_path
        
    name_list = model_args.model_name_or_path.split("_")
    
    perturbed_type = ""
    inoculation_p = 0.0
    for i in range(len(name_list)):
        if name_list[i] == "seed":
            training_args.seed = int(name_list[i+1])
        if name_list[i] == "reverse":
            if name_list[i+1] == "True":
                data_args.reverse_order = True
            else:
                data_args.reverse_order = False
        if name_list[i] == "random":
            if name_list[i+1].strip("/") == "True":
                data_args.random_order = True
            else:
                data_args.random_order = False
        if name_list[i] == "data":
            if len(name_list[i+1].split("-")) > 2:
                perturbed_type = "-".join(name_list[i+1].split("-")[2:])
        if name_list[i] == "inoculation":
            inoculation_p = float(name_list[i+1])
    if "word_s_True" in model_args.model_name_or_path:
        data_args.word_swapping = True
    if "word_freq_s_True" in model_args.model_name_or_path:
        data_args.word_freq_swapping = True
            
    import datetime
    date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)
    if len(model_args.model_name_or_path.split("/")) > 1:
        run_name = "{0}_task_{1}_ft_{2}".format(
            date_time,
            data_args.task_name,
            "_".join(model_args.model_name_or_path.strip("/").split("/")[-1].strip("/").split("_")[1:]),
        )
    else:
        assert False
    training_args.run_name = run_name
    logger.info(f"WANDB RUN NAME: {training_args.run_name}")
    if training_args.do_train:
        training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    elif training_args.do_eval:
        training_args.output_dir = os.path.join(training_args.output_dir, "eval_finetuned_models", run_name)
    elif training_args.do_predict:
        training_args.output_dir = os.path.join(training_args.output_dir, "test_finetuned_models", run_name)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Training/evaluation parameters (model_args) {model_args}")
    logger.info(f"Training/evaluation parameters (data_args) {data_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if perturbed_type == "":
        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.
        #
        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
        elif data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if training_args.do_predict:
                if data_args.test_file is not None:
                    train_extension = data_args.train_file.split(".")[-1]
                    test_extension = data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = data_args.test_file
                else:
                    raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
    else:
        train_file = f"../data-files/{data_args.task_name}-{perturbed_type}"
        logger.info(f"***** Loading pre-loaded datasets from the disk directly! *****")
        raw_datasets = DatasetDict.load_from_disk(train_file)
        
    # Labels
    if data_args.task_name is not None:
        if perturbed_type == "":
            is_regression = data_args.task_name == "stsb"
            if not is_regression:
                label_list = raw_datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            dummy_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
            is_regression = data_args.task_name == "stsb"
            if not is_regression:
                label_list = dummy_datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
    else:
        assert False # NEVER ENTERING HERE!
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    need_resize = False
    if inoculation_p == 0.0:
        # only if we are skipping mid-tunning, this flag can be then set.
        if "token_s_True" in model_args.model_name_or_path:
            data_args.token_swapping = True
        if "reinit_emb_True" in model_args.model_name_or_path:
            data_args.reinit_embeddings = True
        logger.warning(f"***** WARNING: Detected inoculation_p={inoculation_p}; initialize the model and the tokenizer from huggingface. *****")
        # we need to make sure tokenizer is the correct one!
        if "albert-base-v2" in model_args.model_name_or_path:
            model_args.tokenizer_name = "albert-base-v2"
            need_resize = True
        elif "bert-base-uncased" in model_args.model_name_or_path:
            model_args.tokenizer_name = "bert-base-uncased"
            # need_resize = True
        elif "deberta-base" in model_args.model_name_or_path:
            model_args.tokenizer_name =  = "microsoft/deberta-base"
            # need_resize = True
        elif "bert-base-cased" in model_args.model_name_or_path:
            model_args.tokenizer_name = "bert-base-cased"
            need_resize = True
        elif "flaubert_base_cased" in model_args.model_name_or_path:
            model_args.tokenizer_name = "flaubert/flaubert_base_cased"
            need_resize = True
        elif "bert-base-dutch-cased" in model_args.model_name_or_path:
            model_args.tokenizer_name = "GroNLP/bert-base-dutch-cased"
            need_resize = True
        else:
            assert False
        if training_args.do_train:
            if "bert-base-uncased" in model_args.model_name_or_path:
                model_args.model_name_or_path = "bert-base-uncased"
            elif "deberta-base" in model_args.model_name_or_path:
                model_args.model_name_or_path =  = "microsoft/deberta-base"
        
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.do_train:
        if inoculation_p == 1.0 and os.path.isdir(model_args.model_name_or_path):
            if "albert-base-v2" in model_args.model_name_or_path or                 "bert-base-cased" in model_args.model_name_or_path or                 "bert-base-uncased" in model_args.model_name_or_path or                 "deberta-base" in model_args.model_name_or_path or                 "bert-base-dutch-cased" in model_args.model_name_or_path:
                logger.info(f"***** WARNING: Reconfig type_vocab_size for mid-tuned models *****")
                config.type_vocab_size = 2
        if need_resize:
            # we need to rewrite the number of type token a little
            # during pretraining, there are two types for reberta
            # during fine-tuning, i think we are only using one?
            if os.path.isdir(model_args.model_name_or_path):
                pass
            else:
                config.type_vocab_size = 2
    elif training_args.do_eval or training_args.do_predict:
        if "albert-base-v2" in model_args.model_name_or_path or             "bert-base-cased" in model_args.model_name_or_path or             "bert-base-uncased" in model_args.model_name_or_path or             "deberta-base" in model_args.model_name_or_path or             "flaubert_base_cased" in model_args.model_name_or_path or             "bert-base-dutch-cased" in model_args.model_name_or_path:
            
            config.type_vocab_size = 2

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if training_args.do_train:
        if need_resize:
            assert False
            logger.info("***** Replacing the word_embeddings and token_type_embeddings with random initialized values *****")
            # this means, we are finetuning directly with new tokenizer.
            # so the model itself has a different tokenizer, we need to resize.
            model.resize_token_embeddings(len(tokenizer))
            # If we resize, we also enforce it to reinit
            # so we are controlling for weights distribution.
            random_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, 
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir
            )
            # we need to check if type embedding need to be resized as well.
            tokenizer_config = AutoConfig.from_pretrained(
                model_args.tokenizer_name, 
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir
            )
            # IMPORTANT: THIS ENSURES TYPE WILL NOT CAUSE UNREF POINTER ISSUE.
            try:
                random_config.type_vocab_size = tokenizer_config.type_vocab_size
            except:
                random_config.type_vocab_size = 2
            random_model = AutoModelForSequenceClassification.from_config(
                config=random_config,
            )
            random_model.resize_token_embeddings(len(tokenizer))
            replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data.clone()
            model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
            replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
            model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
        if "flaubert_base_cased" in model_args.model_name_or_path and inoculation_p == 1.0:
            # If we resize, we also enforce it to reinit
            # so we are controlling for weights distribution.
            random_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, 
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir
            )
            random_config.type_vocab_size = 2
            random_model = AutoModelForSequenceClassification.from_config(
                config=random_config,
            )
            replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
            replacing_type_embeddings[1] = model.roberta.embeddings.token_type_embeddings.weight.data[0]
            
            # just swap the second one for randomly initialized weights.
            model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
    
    if training_args.do_train:
        if data_args.reinit_avg_embeddings:
            logger.info("***** WARNING: We reinit all embeddings to be the average embedding from the pretrained model. *****")
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir
            )
            avg_embeddings = torch.mean(pretrained_model.roberta.embeddings.word_embeddings.weight.data, dim=0).expand_as(model.roberta.embeddings.word_embeddings.weight.data)
            model.roberta.embeddings.word_embeddings.weight.data = avg_embeddings
            # to keep consistent, we also need to reinit the type embeddings.
            random_model = AutoModelForSequenceClassification.from_config(
                config=config,
            )
            replacing_type_embeddings = random_model.roberta.embeddings.token_type_embeddings.weight.data.clone()
            model.roberta.embeddings.token_type_embeddings.weight.data = replacing_type_embeddings
        elif data_args.reinit_embeddings:
            logger.info("***** WARNING: We reinit all embeddings to be the randomly initialized embeddings. *****")

            if "bert-base-uncased" in model_args.model_name_or_path:
                random_model = AutoModelForSequenceClassification.from_config(config)
                # random_model.resize_token_embeddings(len(tokenizer))
                replacing_embeddings = random_model.bert.embeddings.word_embeddings.weight.data.clone()
                model.bert.embeddings.word_embeddings.weight.data = replacing_embeddings
            elif "deberta-base" in model_args.model_name_or_path:
                random_model = AutoModelForSequenceClassification.from_config(config)
                # random_model.resize_token_embeddings(len(tokenizer))
                replacing_embeddings = random_model.deberta.embeddings.word_embeddings.weight.data.clone()
                model.deberta.embeddings.word_embeddings.weight.data = replacing_embeddings
            elif "roberta" in model_args.model_name_or_path:
                random_model = AutoModelForSequenceClassification.from_config(config)
                # random_model.resize_token_embeddings(len(tokenizer))
                replacing_embeddings = random_model.roberta.embeddings.word_embeddings.weight.data.clone()
                model.roberta.embeddings.word_embeddings.weight.data = replacing_embeddings
        else:
            pass

    if training_args.do_train:
        if data_args.token_swapping:
            logger.info("***** WARNING: We are swapping tokens via embeddings. *****")
            original_embeddings = model.roberta.embeddings.word_embeddings.weight.data.clone()
            g = torch.Generator()
            g.manual_seed(training_args.seed)
            perm_idx = torch.randperm(original_embeddings.size()[0], generator=g)
            swapped_embeddings = original_embeddings.index_select(dim=0, index=perm_idx)
            model.roberta.embeddings.word_embeddings.weight.data = swapped_embeddings

    if data_args.word_swapping:
        assert not data_args.word_freq_swapping
        logger.info("***** WARNING: We are swapping words in the inputs. *****")
        token_frequency_map = json.load(open(data_args.swap_vocab_file))
        wikitext_vocab = list(set(token_frequency_map.keys()))
        # sort so we have consistent map.
        wikitext_vocab.sort()
        wikitext_vocab_copy = copy.deepcopy(wikitext_vocab)
        random.Random(training_args.seed).shuffle(wikitext_vocab_copy)
        word_swap_map = {}
        for i in range(len(wikitext_vocab)):
            word_swap_map[wikitext_vocab[i]] = wikitext_vocab_copy[i]
        # assert word_swap_map["hello"] == "mk14"
    elif data_args.word_freq_swapping:
        assert not data_args.word_swapping
        logger.info("***** WARNING: We are swapping words in the inputs based on similar frequencies. *****")
        token_frequency_map = json.load(open(data_args.swap_vocab_file))
        token_by_length = json.load(open(data_args.swap_token_length_file))
        # TODO: to do a live swap? which maybe really slow!
        word_swap_map = json.load(open(data_args.pre_swap_map_file))
    else:
        # normal cases
        pass
    
    if "bert-base-uncased" in model_args.model_name_or_path:
        assert len(tokenizer) == model.bert.embeddings.word_embeddings.weight.data.shape[0]
    elif "roberta-base" in model_args.model_name_or_path:
        assert len(tokenizer) == model.roberta.embeddings.word_embeddings.weight.data.shape[0]
    elif "deberta" in model_args.model_name_or_path:
        assert len(tokenizer) == model.deberta.embeddings.word_embeddings.weight.data.shape[0]
    logger.info(f"***** Current setups *****")
    logger.info(f"***** model type: {model_args.model_name_or_path} *****")
    logger.info(f"***** tokenizer type: {model_args.tokenizer_name} *****")
    
    def reverse_order(example):
        fields = task_to_keys[data_args.task_name]
        for field in fields:
            if field:
                original_text = example[field]
                original_text = original_text.split(" ")[::-1]
                example[field] = " ".join(original_text)
        return example

    def random_order(example):
        fields = task_to_keys[data_args.task_name]
        for field in fields:
            if field:
                original_text = example[field]
                original_text = original_text.split(" ")
                random.shuffle(original_text)
                example[field] = " ".join(original_text)
        return example
    
    if data_args.reverse_order:
        logger.warning("WARNING: you are reversing the order of your sequences.")
        raw_datasets["train"] = raw_datasets["train"].map(reverse_order)
        validation_name = "validation_matched" if data_args.task_name == "mnli" else "validation"
        raw_datasets[validation_name] = raw_datasets[validation_name].map(reverse_order)
        test_name = "test_matched" if data_args.task_name == "mnli" else "test"
        raw_datasets[test_name] = raw_datasets[test_name].map(reverse_order)

    if data_args.random_order:
        logger.warning("WARNING: you are random ordering your sequences.")
        raw_datasets["train"] = raw_datasets["train"].map(random_order)
        validation_name = "validation_matched" if data_args.task_name == "mnli" else "validation"
        raw_datasets[validation_name] = raw_datasets[validation_name].map(random_order)
        test_name = "test_matched" if data_args.task_name == "mnli" else "test"
        raw_datasets[test_name] = raw_datasets[test_name].map(random_order)
    # we don't care about test set in this script?

    if data_args.word_swapping or data_args.word_freq_swapping:
        logger.warning("WARNING: performing word swapping.")
        # we need to do the swap on the data files.
        # this tokenizer helps you to get piece length for each token
        modified_tokenizer = ModifiedBertTokenizer(
            vocab_file="../data-files/bert_vocab.txt")
        modified_basic_tokenizer = ModifiedBasicTokenizer()
        raw_datasets["train"] = raw_datasets["train"].map(partial(random_corrupt, 
                                                       data_args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))
        validation_name = "validation_matched" if data_args.task_name == "mnli" else "validation"
        raw_datasets[validation_name] = raw_datasets[validation_name].map(partial(random_corrupt, 
                                                       data_args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))
        test_name = "test_matched" if data_args.task_name == "mnli" else "test"
        raw_datasets[test_name] = raw_datasets[test_name].map(partial(random_corrupt, 
                                                       data_args.task_name,
                                                       modified_basic_tokenizer, 
                                                       word_swap_map))
    if data_args.task_name == "mnli":
        # process the other part.
        if data_args.reverse_order:
            validation_name = "validation_mismatched" if data_args.task_name == "mnli" else "validation"
            raw_datasets[validation_name] = raw_datasets[validation_name].map(reverse_order)
            test_name = "test_mismatched" if data_args.task_name == "mnli" else "test"
            raw_datasets[test_name] = raw_datasets[test_name].map(reverse_order)
        if data_args.random_order:
            validation_name = "validation_mismatched" if data_args.task_name == "mnli" else "validation"
            raw_datasets[validation_name] = raw_datasets[validation_name].map(random_order)
            test_name = "test_mismatched" if data_args.task_name == "mnli" else "test"
            raw_datasets[test_name] = raw_datasets[test_name].map(random_order)
        if data_args.word_swapping or data_args.word_freq_swapping:
            validation_name = "validation_mismatched" if data_args.task_name == "mnli" else "validation"
            raw_datasets[validation_name] = raw_datasets[validation_name].map(partial(random_corrupt, 
                                                           data_args.task_name,
                                                           modified_basic_tokenizer, 
                                                           word_swap_map))
            test_name = "test_mismatched" if data_args.task_name == "mnli" else "test"
            raw_datasets[test_name] = raw_datasets[test_name].map(partial(random_corrupt, 
                                                           data_args.task_name,
                                                           modified_basic_tokenizer, 
                                                           word_swap_map))

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        
        # the fixed output file.
        eval_metrics_output_path = "../glue_evaluation_results.csv"
        my_file = Path(eval_metrics_output_path)
        if my_file.is_file():
            pass
        else:
            # we need to write the header.
            with open(eval_metrics_output_path, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([
                    "glue_task", "split", "model", "learning_rate", "tokenizer", "midtuning",
                    "galactic_shift", "reinit_embedding", "reverse_order", "random_order", 
                    "token_swap", "word_swap", "metrics", "performance"
                ])
            
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
            for m in task_to_metrics[task]:
                with open(eval_metrics_output_path, mode='a') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([
                        task, "eval", out_model, out_lr, out_tokenizer_name, out_midtuning, 
                        out_galactic_shift, out_reinit_embedding, out_reverse, out_random, 
                        out_token_s, out_word_s, m, metrics[f"eval_{m}"],
                    ])
            
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


# In[ ]:


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:





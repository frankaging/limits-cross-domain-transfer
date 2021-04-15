# Identifying the Limits of Cross-Domain Knowledge Transfer for Pretrained Models

There is growing evidence that pretrained language models improve task-specific fine-tuning not just for the languages seen in pretraining, but also for new languages and even non-linguistic data. What is the source of this surprising cross-domain transfer? We offer a partial answer via a systematic exploration of how much transfer occurs when models are denied any information about word identity via random scrambling.

## Contents

* [Citation](#Citation)
* [Quick start](#quick-start)
* [License](#license)

## Citation

[Zhengxuan Wu](http://zen-wu.social), [Nelson F. Liu](https://cs.stanford.edu/~nfliu/), [Christopher Potts](http://web.stanford.edu/~cgpotts/). 2021. [Identifying the Limits of Cross-Domain Knowledge Transfer for Pretrained Models](https://arxiv.org/abs/TO_APPEAR). Ms., Stanford University.

```stex
  @article{wu-etal-2021-identify,
    title={Identifying the Limits of Cross-Domain Knowledge Transfer for Pretrained Models},
    author={Wu, Zhengxuan and Liu, Nelson F. and Potts, Christopher},
    journal={},
    url={},
    year={2021}}
```

## Quick start

### Install Requirements
You have to download modules required by [HuggingFace Transformer](https://github.com/huggingface/transformers). Note that you also have to install all the dependecies needed for running HuggingFace as well. For future, we will add a auto install script here so you don't have to worry about it (in the most cases).

### Scramble Inputs
One important experiment we ran was to finetune BERT with systematically scrambled English sentences. This process can be found in the notebook via running,
```bash
cd code/
jupyter notebook
```
In `vocab_mismatch.ipynb` notebook, it will work you through how we generate scrambled datasets.

### BERT Model Training
We largely use [HuggingFace Transformer](https://github.com/huggingface/transformers) for our model training to ensure reproducibility. There are two main scripts that help you run both sequence classification and labeling tasks with some useful options.

Here is an example command to run sequence classification:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sentence_classification.py \
--run_name sst-tenary \
--task_name sst3 \
--inoculation_data_path ../data-files/sst-tenary \
--model_type bert-base-uncased \
--output_dir ../sst-tenary-result/ \
--max_seq_length 128 \
--learning_rate 2e-5 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--metric_for_best_model Macro-F1 \
--greater_is_better \
--is_tensorboard \
--logging_steps 10 \
--eval_steps 10 \
--seed 42 \
--load_best_model_at_end \
--inoculation_step_sample_size 1.00 \
--num_train_epochs 3 \
--inoculation_patience_count 5 \
--save_total_limit 3
```

Here is an example command to run sequence labeling:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_token_classification.py \
--run_name conll2003 \
--task_name conll2003 \
--inoculation_data_path conll2003 \
--token_type ner_tags
--model_type bert-base-uncased \
--output_dir ../conll2003-result/ \
--max_seq_length 128 \
--learning_rate 2e-5 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 128 \
--metric_for_best_model f1 \
--greater_is_better \
--is_tensorboard \
--logging_steps 10 \
--eval_steps 10 \
--seed 42 \
--load_best_model_at_end \
--inoculation_step_sample_size 1.00 \
--num_train_epochs 3 \
--inoculation_patience_count 5 \
--save_total_limit 3 \
--n_layer_to_finetune 1
```

Here is an example command to run evaluation:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py \
--task_name classification \
--model_path ../saved-models/sst-tenary-result/pytorch_model.bin \
--model_type bert-base-uncased \
--cache_dir ../tmp/ \
--max_seq_length 128 \
--per_device_eval_batch_size 64 \
--data_path ../data-files/sst-tenary/sst-tenary-test.tsv
```

#### Model Training Options
As you can see, we devloped our own wrapper outside HuggingFace script. You have different options to setup your training scripts. For example `--inoculation_step_sample_size` can control how much in proportion you want to use to train the model. `--scramble_proportion` allows you to see the ordering effect in case you want to see how model perform when fine-tuning with scrambled datasets. `--is_tensorboard` will allow you log results to [Weights & Biases](https://wandb.ai/home). `--inoculation_patience_count` let you control for patient steps. For full support, you can look into the script. Since HuggingFace is constantly updating their scripts, you may need to adapt the codebase! `--n_layer_to_finetune` let you control how many layers you want to fine-tune. `--no_pretrain` will no load BERT pretrained weights. `--model_type` or `--model_path` helps you with loading models from HuggingFace or from local drive.


### Other Model Training
Other than BERT, we also offer training for other models.

#### BoW, CRFs and Random Classifers
BoW and Random model is included in the notebook as they are easy to train with CPU at `run_bow_classifier.ipynb` and `run_crf_classifier.ipynb`.

#### Pretraining BERT from Scratch with Scrambled Data
Yes, we also allow you to do this. We tried this but not included in the paper. To pretrain a BERT, you can use our wrapper as:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python run_pretrain_bert.py \
--model_type bert \
--train_file ../data_files/wikitext-15M \
--do_train \
--do_eval \
--output_dir ../256-bert-base-uncased-wikitext-15M-results/ \
--cache_dir ../.huggingface_cache/ \
--num_train_epochs 40 \
--seed 42 \
--max_seq_length 256 \
--run_name 256-mlm-bert-base-uncased-wikitext-15M-results \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--learning_rate 2e-5 \
--evaluation_strategy steps \
--eval_steps 500 \
--logging_steps 50 \
--line_by_line \
--pad_to_max_length
```
You can replace `--train_file` with the scrambled dataset file. And pretrain your MLMs!

#### LSTM + GloVe
We use an open source training script at here [LSTM-GloVe](https://github.com/frankaging/BERT_LRP/). To train a LSTM model, you can use:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier.py \
--model_type LSTMSequenceLabeling \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 512 \
--eval_batch_size 512 \
--learning_rate 1e-3 \
--num_train_epochs 200 \
--seed 123 \
--task_name CONLL_2003 \
--data_dir ../../pretrain-data-distribution/data-files/conll2003-corrupted-matched/ \
--vocab_file ../models/LSTM/vocab.txt \
--output_dir ../results/CONLL_2003-LSTM-scratch-matched/
```
You have two options `--embed_file` to specify if you want to load pretrained embeddings for the vocab file. And you may change your model type to `--model_type LSTM` for sequence classification tasks.

#### Making Our TensorBoard Public
We are hoping to release a set of tensorboard to public, so to ensure reproducibility! Here is an example:

[SST-3 BERT Training Results](https://wandb.ai/wuzhengx/sst3_bertonomy)

Stayed tuned. We need to go through them and name them correctly.

## License

This repo has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).





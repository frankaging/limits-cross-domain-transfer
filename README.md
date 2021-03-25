### Introduction
This is the repo for investigating the Transferability of BERT model.

### Install Requirements
You have to download modules required by [HuggingFace Transformer](https://github.com/huggingface/transformers). Note that you also have to install all the dependecies needed for running HuggingFace as well. For future, we will add a auto install script here so you don't have to worry about it (in the most cases).

### Scramble Inputs
One important experiment we ran was to finetune BERT with systematically scrambled English sentences. This process can be found in the notebook via running,
```bash
cd code/
jupyter notebook
```
In this notebook, it will work you through how we generate scrambled datasets.

### Model Training


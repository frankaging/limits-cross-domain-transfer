#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import os


# In[4]:


eval_method = "do_eval"
eval_model_path = "../finetuned_models"
wandb_panel = "ICLR_GLUE_eval"


# In[ ]:


for path in glob(f"{eval_model_path}/*/"):
    print(f"generating results for path at: {path}")
    if "mnli" in path or "qqp" in path:
        cmd = f"CUDA_VISIBLE_DEVICES=8 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 16               --output_dir ../eval_finetuned_models"
    else:
        cmd = f"CUDA_VISIBLE_DEVICES=8 python run_glue.py               --model_name_or_path {path}               --{eval_method} --per_device_eval_batch_size 32               --output_dir ../eval_finetuned_models"
    print(f"starting command")
    os.system(cmd)


# In[ ]:


print("**********")
print("*  Test  *")
print("**********")
# verification steps.
all_records = set([])
path_record_map = {}
for path in glob(f"{eval_model_path}/*/"):
    record = ("_".join(path.strip("/").split("/")[-1].split("_")[1:]))
    all_records.add(record)
    path_record_map[record] = path
    
assert len(path_record_map) == len(all_records)

import wandb
api = wandb.Api()
runs = api.runs(f"wuzhengx/{wandb_panel}")

all_wandb_records = []
for run in runs:
    run_name = run.name
    run_name = "_".join(run_name.split("_")[4:])
    all_wandb_records.append(run_name)

print("**********")
print("Rerun following experiments:")
count = 0 
for r in all_records:
    if r not in all_wandb_records:
        print(path_record_map[r])
        count += 1
print("**********")
if count == 0:
    print("Test Result: Passed.")
print("**********")


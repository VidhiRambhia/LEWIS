# **LEWIS (LayEr WIse Sparsity) - A Training-Free Guided Model Merging Approach**

**Accepted at ICLR 2025 Workshop on Sparsity in LLMs (SLLM)** - [Official Website](https://www.sparsellm.org/)  
**Paper:** [arXiv:2503.03874](https://www.arxiv.org/abs/2503.03874)  

## **Overview**

As specialized large language models (LLMs) become increasingly prevalent, **model merging** methods are being used to combine them into a single multi-task model **without requiring additional data or training**. However, existing approaches struggle when the objective is to **improve a model's performance on specific task benchmarks** after merging.

In this work, we propose **LEWIS (Layer Wise Sparsity)**—a **guided model-merging framework** that uses **activation-based layer importance** to dynamically adjust **layer-wise task-vector sparsity** during the merge process. LEWIS leverages a **calibration dataset** to prioritize critical layers during the **task-vector pruning** step, preserving essential **task-specific knowledge** while optimizing performance on targeted benchmarks.

---

## **Installation & Setup**

### **1. Create and Activate the Conda Environment**
```bash
conda create -n lewis python=3.10 -y
conda activate lewis
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
pip install -e mergekit
```

### **3. Clone the Repository**
```bash
git clone https://github.com/VidhiRambhia/LEWIS
cd LEWIS
```

---

## **Usage**


### **1. Getting LEWIS Importance Scores and storing them in a dataframe**
```bash
mkdir -p "output/gemma_2b/delta_dfs"
python3 lewis_importance.py \
        --config "configs/gemma_2b.yaml" \
        --out_dir "output/gemma_2b/delta_dfs"
```

### **1. Generate Merge Configurations using LEWIS Importance Scores**
```bash
mkdir -p "output/gemma_2b/merge_configs"
python3 lewis_merge.py \
   --delta_dfs_in_dir "output/gemma_2b/delta_dfs" \ #delta dfs will be generated above
   --config_file "configs/gemma_2b.yaml" \
   --out_dir "output/gemma_2b/merge_configs" \
   --gamma 0.3 \
   --epsilon 0.8 \
   --merge_method "ties"
```

### **2. Perform Model Merging**
```bash
mkdir -p "output/gemma_2b/merged_models"
python3 utils/merger.py \
    --output_path "output/gemma_2b/merged_models" \
    --merge_config "output/gemma_2b/merge_configs/gemma_2b_0.3_0.8_generated_ties_merge_config.yaml" \ #generated above
    --copy_tokenizer True \
    --lazy_unpickle False \
    --low_cpu_memory False \
    --trust_remote_code True \
    --allow_crimes True
```

---

## **Citation**
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{
chopra2025lewis,
title={{LEWIS} (LayEr {WI}se Sparsity) - A Training Free Guided Model Merging Approach},
author={Hetarth Chopra and Vidhi Rambhia and Vikram S. Adve},
booktitle={Sparsity in LLMs (SLLM): Deep Dive into Mixture of Experts, Quantization, Hardware, and Inference},
year={2025},
url={https://openreview.net/forum?id=FPzHR354PK}
}
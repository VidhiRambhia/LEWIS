
import torch.nn as nn 
import torch
from datasets import load_dataset
import argparse
import tqdm
import os 
import gc
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from utils.WrappedGPT import WrappedGPT

def get_mbpp(nsamples, seed, seqlen, tokenizer):
    print("Downloading the dataset")
    dataset = load_dataset("google-research-datasets/mbpp", name='full', split='train')
    dataset = dataset.shuffle(seed=seed).take(nsamples)

    trainloader = []
    for sample in dataset:
        code = sample["text"] + " " + sample["code"]
        # Use tokenizer to pad and truncate the input to 'seqlen'
        inp = tokenizer(
            code,
            return_tensors="pt",
            max_length=seqlen,
            padding='max_length',
            truncation=True
        )
        tar = inp.input_ids.clone()
        tar[:, :-1] = -100  # Set target sequence
        trainloader.append((inp.input_ids, tar))

    return trainloader, None

def get_gsm8k(nsamples, seed, seqlen, tokenizer):
    print("Downloading the GSM8K dataset")
    # Load the GSM8K dataset
    dataset = load_dataset("openai/gsm8k", name='main', split='train')  # Change to 'test' if needed
    dataset = dataset.shuffle(seed=seed).select(range(nsamples))  # Select a subset of samples

    trainloader = []
    for sample in dataset:
        # Concatenate question and answer
        qa = sample["question"] + " " + sample["answer"]
        # Use tokenizer to pad and truncate the input to 'seqlen'
        inp = tokenizer(
            qa,
            return_tensors="pt",
            max_length=seqlen,
            padding='max_length',
            truncation=True
        )
        tar = inp.input_ids.clone()
        tar[:, :-1] = -100  # Mask all but the last token for target sequence
        trainloader.append((inp.input_ids, tar))

    return trainloader, None

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): Dataset name.
        nsamples (int, optional): Number of samples. Defaults to 128.
        seed (int, optional): Seed for reproducibility. Defaults to 0.
        seqlen (int, optional): Sequence length. Defaults to 2048.
        tokenizer (object, optional): Tokenizer object. Defaults to None.

    Returns:
        trainloader (list): List of training samples.
        testenc (object): Test dataset encoding.
    """
    if name == "mbpp":
        return get_mbpp(nsamples, seed, seqlen, tokenizer)
    elif name == "gsm8k":
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError("Invalid dataset name")

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = True
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_activation_norm(sparsity_ratio, model, tokenizer, dataset, device=torch.device("cuda:0"), nsamples=15):
    use_cache = model.config.use_cache 
    model.config.use_cache = True 

    print("Loading calibdation data")
    dataloader, _ = get_loaders(dataset,nsamples=nsamples,seed=42,seqlen=model.seqlen,tokenizer=tokenizer)
    print("Dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    W_metric_norm_dict= {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        wrapped_layers = {}
        print("Wrapping the layers")
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        for h in handles:
            h.remove()

        for name in tqdm.tqdm(subset):
            print(f"Getting activation norm for layer {i} name {name}")
            print("Scaler Row shape = " , torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))).shape)
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            W_metric_norm = torch.norm(W_metric)

            W_metric_norm_dict[f'Layer_{name}_{i}']= W_metric_norm.item()

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
                
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    model.to('cpu')
    del inps, outs, attention_mask, position_ids, layers, subset, wrapped_layers
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return W_metric_norm_dict

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto' )
    print(f"loading llm model {model_name}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Add this line
    device = torch.device("cuda")
    model.seqlen = 512
    return model, tokenizer, device

def calculate_layer_norms(model_name):
    model, tokenizer, device = load_model(model_name)
    layers = model.model.layers
    Norm_dict = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        wrapped_layers = {}
        print("Wrapping the layers")
        for name in subset:
            print(name)
            wrapped_layers[name] = WrappedGPT(subset[name])

        for name in subset:
            param_vector = subset[name].weight.data.cpu().flatten().numpy()
            norm = torch.norm(subset[name].weight.data.cpu())
            Norm_dict[f'Layer_{name}_{i}'] = norm.item()

    torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return Norm_dict

def post_process(initial_df, components):
    """
    Post-process the DataFrame to prepare it for delta computation.

    Args:
        initial_df (pd.DataFrame): Initial concatenated DataFrame.
        components (list): List of components to consider.

    Returns:
        pd.DataFrame: Final processed DataFrame.
    """
    df = initial_df.copy()

    # Extract 'layer_number' from 'layer_name'
    df['layer_number'] = df['layer_name'].str.extract(r'_(\d+)$').astype(int)

    # Extract 'component' from 'layer_name'
    df['component'] = df['layer_name'].str.extract(r'\.(\w+)_\d+$')

    # Melt the dataframe to long format
    id_vars = ['layer_number', 'component']
    value_vars = [col for col in df.columns if col not in id_vars + ['layer_name']]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='model',
        value_name='value'
    )

    # Create a new column combining component and model
    df_melted['component_model'] = df_melted['component'] + '_' + df_melted['model']

    # Pivot the dataframe to wide format
    df_pivoted = df_melted.pivot_table(
        index='layer_number',
        columns='component_model',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Reorder the columns dynamically
    models = df_melted['model'].unique()
    columns_order = ['layer_number'] + [
        f"{comp}_{model}" for model in models for comp in components
    ]
    df_final = df_pivoted.reindex(columns=columns_order)

    return df_final

def compute_deltas(df_final, components, base_model_type):
    """
    Compute delta values between the base model and other models.

    Args:
        df_final (pd.DataFrame): Processed DataFrame from post_process.
        components (list): List of components to consider.
        base_model_type (str): The base model type for delta comparison.

    Returns:
        dict: Dictionary containing delta DataFrames for each model compared to the base model.
    """
    delta_dfs = {}
    models = [
        col.split('_')[-1] for col in df_final.columns if col != 'layer_number'
    ]
    models = list(set(models))
    models.remove(base_model_type)

    for model in models:
        df_delta = pd.DataFrame()
        for component in components:
            base_col = f"{component}_{base_model_type}"
            model_col = f"{component}_{model}"
            delta_col = f"{component}_delta_{model}_{base_model_type}"
            df_delta[delta_col] = (df_final[model_col] - df_final[base_col]).abs()
            # Normalize
            total = df_delta[delta_col].sum()
            if total != 0:
                df_delta[delta_col] /= total
            # Clip values between 0 and 1
            df_delta[delta_col] = df_delta[delta_col].clip(0, 1)
        delta_dfs[model] = df_delta

    return delta_dfs

def run_lewis(model_info, components, sparsity_ratio, nsamples, dataset):
    """
    Main function to process models and compute W_metric norms.

    Args:
        model_info (list of dict): List containing model names and their types.
        components (list): List of components to consider.
        sparsity_ratio (float): Sparsity ratio for pruning.

    Returns:
        pd.DataFrame: Concatenated DataFrame of W_metric norms.
    """
    W_metric_norms = []
    for model in model_info:
        model_name = model['name']
        model_type = model['type']
        model_obj, tokenizer, device = load_model(model_name)
        W_metric_norm = get_activation_norm(
            sparsity_ratio=sparsity_ratio,
            model=model_obj,
            tokenizer=tokenizer,
            device=device,
            nsamples=nsamples,
            dataset=dataset
        )
        W_metric_norms.append({
            'name': model_name,
            'type': model_type,
            'norm': W_metric_norm
        })
        del model_obj
    
    # Create DataFrames for each model
    w_metric_pandas_dict = {}
    for entry in W_metric_norms:
        model_type = entry['type']
        w_metric_norm = entry['norm']
        df = pd.DataFrame({
            'layer_name': list(w_metric_norm.keys()),
            model_type: list(w_metric_norm.values())
            
        })
        w_metric_pandas_dict[model_type] = df
    
    # Merge DataFrames on 'layer_name'
    df_list = list(w_metric_pandas_dict.values())
    w_metric_pandas_concat = df_list[0]
    for df in df_list[1:]:
        w_metric_pandas_concat = pd.merge(
            w_metric_pandas_concat, df, on='layer_name', how='outer'
        )
    df_final = post_process(w_metric_pandas_concat, components)
    compute_deltas(df_final, components, base_model_type)
    return delta_dfs

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a configuration file based on a YAML config.")
    parser.add_argument('--config', type=str, help="The path to the YAML config file")
    parser.add_argument('--out_dir', type=str, help="The path to the output directory where the generated YAML config file will be saved")
    args = parser.parse_args()

    with open(f"{args.config}", 'r') as file:
            print(file)
            config_data = yaml.safe_load(file)

    model_info = config_data.get('model_info', [])
    components = config_data.get('components', [])
    base_model_type = config_data.get('base_model_type', 'base')
    base_model_name = config_data.get('base_model_name', '')
    target_model_names = config_data.get('target_model_names', {})
    global_parameters = config_data.get('global_parameters', {})
    prefix_mapping = config_data.get('prefix_mapping', {})
    base_name = os.path.splitext(os.path.basename(args.config))[0]
    sparsity_ratio = config_data.get('sparsity_ratio', 0)
    nsamples = config_data.get('nsamples', 15)
    dataset = config_data.get('dataset', "mbpp")

    # Run main processing
    delta_dfs = run_lewis(model_info, components, sparsity_ratio=sparsity_ratio, nsamples=nsamples, dataset=dataset)
    
    delta_output_dir = f"{args.out_dir}/{os.path.splitext(args.config)[0]}_{dataset}_dataset"
    
    ensure_directory_exists(delta_output_dir)

    # Save delta_dfs to individual CSV files in the output directory
    for model_name, df_delta in delta_dfs.items():
        delta_output_path = f"{delta_output_dir}/{model_name}_delta.csv"
        df_delta.to_csv(delta_output_path, index=False)
        print(f"Saved delta DataFrame for {model_name} to {delta_output_path}")

    print("All delta dataFrames have been saved.")
import argparse
import os 
import pandas as pd
import yaml

def load_delta_dfs(output_directory):
    """
    Loads all the CSV files from the output directory into a dictionary of DataFrames.

    Args:
        output_directory (str): The path to the directory where the delta DataFrames are saved.

    Returns:
        dict: A dictionary where the keys are the model names (from the CSV filenames) and the values are the loaded DataFrames.
    """
    delta_dfs = {}
    for filename in os.listdir(output_directory):
        if filename.endswith("_delta.csv"):
            model_name = filename.replace('_delta.csv', '')
            df = pd.read_csv(os.path.join(output_directory, filename))
            delta_dfs[model_name] = df
            print(f"Loaded DataFrame for {model_name} from {filename}")

    return delta_dfs

def create_sparsity_values(df_delta, components, model_suffix, prefix_mapping, gamma, epsilon):
    """
    Create a density values dictionary from the delta DataFrame.

    Args:
        df_delta (pd.DataFrame): Delta DataFrame for a specific model.
        components (list): List of components to consider.
        model_suffix (str): Suffix of the model used in delta computation.
        prefix_mapping (dict): Mapping from components to their full names.
        **kwargs: Optional arguments, including:
            - min_scale (float): Minimum scale value (default 0.5).
            - max_scale (float): Maximum scale value (default 0.8).
            - top_k (int): Top-k percentage for retaining values (default 1, meaning scaling).

    Returns:
        dict: Dictionary of density values.
    """
    # Set default values for optional arguments
    density_values = {}
    for component in components:
        delta_col = f"{component}_delta_{model_suffix}"
        full_component_name = prefix_mapping.get(component, component)
        for index, value in df_delta[delta_col].items():
            key = f"model.layers.{index}.{full_component_name}"
            density_values[key] = round(float(value), 5)
            
    values = list(density_values.values())
    min_val = min(values)
    max_val = max(values)

    # Handle the case where all values are the same
    if min_val == max_val:
        return {key: (gamma + epsilon) / 2 for key in density_values}

    # Scaling values to the provided range
    scaled_values = {
        key: gamma + (val - min_val) / (max_val - min_val) * (epsilon - gamma)
        for key, val in density_values.items()
    }
    return scaled_values


def generate_config_helper(
    base_model_name,
    density_values_dict,
    global_parameters,
    merge_method,
    apply_lewis_to_weights
):
    """
    Generate a configuration dictionary for YAML output.

    Args:
        base_model_name (str): Name of the base model.
        target_model_names (list): Names of the target models.
        density_values_dict (dict): Dictionary of density values for each target model.
        global_parameters (dict): Global parameters for the config.

    Returns:
        dict: Configuration dictionary.
    """
    # Define the models list
    models = [{'model': base_model_name}]
    if apply_lewis_to_weights:
        for target_model_name, density_values in density_values_dict.items():
            models.append({
                'model': target_model_name,
                'parameters': {
                    'density': [
                        {'filter': key, 'value': value}
                        for key, value in density_values.items()
                    ] + [{'value': 1}],
                    'weight': [
                        {'filter': key, 'value': value}
                        for key, value in density_values.items()
                    ] + [{'value': 1}],
                }
            })
    else:
        for target_model_name, density_values in density_values_dict.items():
            models.append({
                'model': target_model_name,
                'parameters': {
                    'density': [
                        {'filter': key, 'value': value}
                        for key, value in density_values.items()
                    ] + [{'value': 1}],
                    'weight': [{'value': 1}]
                }
            })
        
    # Define the config dictionary
    config = {
        'models': models,
        'merge_method': merge_method,
        'base_model': base_model_name,
        'parameters': global_parameters,
        'dtype': 'bfloat16',
        'tokenizer_source': 'union'
    }

    return config

def generate_merge_config(config_file, delta_dfs_in_dir, out_dir, gamma, epsilon, merge_method, apply_lewis_to_weights):
    with open(f"{config_file}", 'r') as file:
        config_data = yaml.safe_load(file)
    components = config_data.get('components', [])
    base_model_type = config_data.get('base_model_type', 'base')
    base_model_name = config_data.get('base_model_name', '')
    target_model_names = config_data.get('target_model_names', {})
    global_parameters = config_data.get('global_parameters', {})
    prefix_mapping = config_data.get('prefix_mapping', {})
    delta_dfs = load_delta_dfs(delta_dfs_in_dir)
    density_values_dict = {}
    for model_type, target_model_name in target_model_names.items():
        model_suffix = f"{model_type}_{base_model_type}"
        df_delta = delta_dfs.get(model_type)
        if df_delta is not None:
            density_values = create_sparsity_values(df_delta, components, model_suffix, prefix_mapping, gamma, epsilon)
            density_values_dict[target_model_name] = density_values
        
        config = generate_config_helper(base_model_name, density_values_dict, global_parameters, merge_method, apply_lewis_to_weights)
        # Extract base name from config_file without directory
        config_base_name = os.path.splitext(os.path.basename(config_file))[0]
        output_filename = os.path.join(out_dir, f"{config_base_name}_{gamma}_{epsilon}_generated_{merge_method}_merge_config.yaml")
        # output_filename = f"{out_dir}/{os.path.splitext(config_file)[0]}_{gamma}_{epsilon}_generated_{merge_method}_merge_config.yaml"
        with open(output_filename, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
        print(f"The merge config file '{output_filename}' has been generated successfully.")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate a configuration file based on a YAML config.")

    # Add arguments for the input directory, YAML file, and output directory
    parser.add_argument('--delta_dfs_in_dir', type=str, help="The path of the delta dfs directory")
    parser.add_argument('--config_file', type=str, help="The path to the YAML config file")
    parser.add_argument('--out_dir', type=str, help="The path to the output directory where the generated YAML config file will be saved") 
    parser.add_argument('--gamma', type=float, help="Gamma value")
    parser.add_argument('--epsilon', type=float, help="Epsilon value")
    parser.add_argument('--merge_method', type=str, default="ties", help="Merge Kit merge method")
    parser.add_argument('--apply_lewis_to_weights', action="store_true", help="Apply Lewis Importance during weights merging as well")
        
    # Parse the arguments
    args = parser.parse_args()
    generate_merge_config(args.config_file, args.delta_dfs_in_dir, args.out_dir, args.gamma, args.epsilon, args.merge_method, args.apply_lewis_to_weights)
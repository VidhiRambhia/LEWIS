import torch
import yaml
import argparse
# from mergekit.mergekit import *
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

def main(args):
    with open(args.merge_config, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
        print("Merge configuration loaded and validated.")
    print(merge_config)
    print("Running merge...")
    run_merge(
        merge_config,
        out_path=args.output_path,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=args.copy_tokenizer,
            lazy_unpickle=args.lazy_unpickle,
            low_cpu_memory=args.low_cpu_memory,
            trust_remote_code=args.trust_remote_code,
            allow_crimes=args.allow_crimes
        ),
    )
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models with the given configuration.")
    parser.add_argument('--output_path', type=str, help='Folder to store the result in')
    parser.add_argument('--merge_config', type=str, help='Merge configuration file')
    parser.add_argument('--copy_tokenizer', type=bool, default=True, help='Whether to copy the tokenizer')
    parser.add_argument('--lazy_unpickle', type=bool, default=False, help='Experimental low-memory model loader')
    parser.add_argument('--low_cpu_memory', type=bool, default=False, help='Enable if you have more VRAM than RAM+swap')
    parser.add_argument('--trust_remote_code', type=bool, default=True, help='Trust remote code')
    parser.add_argument('--allow_crimes', type=bool, default=True, help='Allow crimes')
    
    args = parser.parse_args()
    main(args)


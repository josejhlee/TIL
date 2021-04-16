import os
import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--output_folder', type=str, default="output_img")
    args = parger.parse_args()

    os.mkdir(args.output_folder, exist_ok=True)

    generator = torch.load(args.model, map_lacation=torch.device("cuda:0" if self.use_cuda else "cpu"))
    gnerator.eval()
    output_folder = args.output_folder

    img = generator()

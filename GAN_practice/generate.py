import os
import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default="./save_model/model_00000.pytorch")
    parser.add_argument('--output_folder', type=str, default="output_img")
    parser.add_argument('--sample_num', type=int, default=64)
    args = parger.parse_args()

    os.mkdir(args.output_folder, exist_ok=True)
    checkpoint = torch.load(args.model, map_lacation=torch.device("cuda:0" if self.use_cuda else "cpu"))
    generator = load_state_dict(checkpoint['generator'])
    gnerator.eval()

    z = torch.randn(args.sample_num, 100, 1, 1,device=torch.device("cuda:0" if self.use_cuda else "cpu"))
    img = generator(z)
    save_image(img,os.path.join(args.output_folder,'result.png'),normalize=True)

    """
    for i in range(args.sample_num):
        img_one = img[i]
        save_image(img_one,os.path.join(args.output_folder,'result_%06d.png' % i))
    """

    

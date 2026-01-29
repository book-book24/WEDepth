import os
import json
import glob
import cv2
import matplotlib
import argparse
import numpy as np
import torch
from module.Model import WEDepth
import torch


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric Depth Estimation')
    parser.add_argument("--config-file", type=str, default="./configs/kitti.json")
    parser.add_argument('--img-path', type=str, default='../ddad/ddad_val.txt')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis')
    parser.add_argument('--load-from', type=str, default='./best_d1_KITTI0723.pth')
    parser.add_argument('--dataset', type=str, default='DDADDataset')

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = WEDepth(config)
    w = torch.load(args.load_from, map_location='cpu')
    w = remove_module_prefix(w)

    model.load_state_dict(w)
    model = model.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)
    # cmap = matplotlib.colormaps.get_cmap('Spectral')
    # cmap = matplotlib.colormaps.get_cmap('inferno_r')
    cmap = matplotlib.colormaps.get_cmap('plasma')

    for k, filename in enumerate(filenames):

        img_name = filename.strip().split(" ")[0]
        gt_name = filename.strip().split(" ")[1]
        print(f'Progress {k + 1}/{len(filenames)}: {img_name}')


        raw_image = cv2.imread(img_name)


        if args.dataset == "KITTIDataset":
            height_start, width_start = int(raw_image.shape[0] - 352), int(
                (raw_image.shape[1] - 1216) / 2)
            height_end, width_end = height_start + 352, width_start + 1216
            raw_image = raw_image[height_start:height_end, width_start:width_end]



        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        depth = model.infer(image=raw_image, mod=args.dataset)


        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)

        if args.dataset == "NYUDataset":

            depth = depth[45:471, 41:601]
        elif args.dataset == "KITTIDataset":
            depth = depth[50:,...]


        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(img_name))[0] + '.png')

        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(output_path, combined_result)

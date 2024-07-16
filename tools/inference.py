import os
import glob
import math

import pandas as pd
from PIL import Image
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from glsim.other_utils.build_args import parse_inference_args
from glsim.data_utils.build_transform import build_transform

from train import build_environment
from heatmap import make_heatmaps, inverse_normalize


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

    # if path is a file
    if os.path.isfile(args.images_path):
        if os.path.splitext(args.images_path)[1] in ('.txt', '.csv'):
            df = pd.read_csv(args.images_path)
            print('Total image files', len(df))
            return df['dir'].tolist()

        elif any([t.replace('*', '') in os.path.splitext(args.images_path)[1] for t in types]):
            return [args.images_path]

    # else if directory
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files', len(files_all))
    return files_all


def prepare_inference(args):
    model, _, _, _, _, _, _ = build_environment(args)
    model.eval()

    if args.results_inference:
        if args.dynamic_anchor:
            model_name = f'{args.model_name}_{args.anchor_size}'
        else:
            model_name = f'{args.model_name}'
        args.results_dir = os.path.join(args.results_inference, f'{model_name}')
        os.makedirs(args.results_dir, exist_ok=True)

    transform = build_transform(args=args, split='test')

    # Load class names
    fp = os.path.join(args.dataset_root_path, args.df_classid_classname)

    if os.path.isfile(fp):
        dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()
    else:
        dic_classid_classname = None

    return model, transform, dic_classid_classname


def save_crops(images_og, images_crops, fp, image_size, resize_og=False,
                save_crops_only=False, norm_custom=False):
    fp = fp.replace('.png', '_crops.png')

    with torch.no_grad():
        if images_crops is not None:
            images_crops = images_crops.reshape(3, image_size, -1)

        if save_crops_only and images_crops is not None:
            samples = inverse_normalize(images_crops.data, norm_custom)
        else:
            if resize_og:
                images_og = F.interpolate(images_og, size=(image_size, image_size), mode='bicubic')
            images_og = images_og.reshape(3, image_size, image_size)

            if images_crops is not None:
                images = torch.cat((images_og, images_crops), dim=2)
            else:
                images = images_og
            samples = inverse_normalize(images.data, norm_custom)

        save_image(samples, fp, nrow=1)
        print(f'Saved file to : {fp}')
    return 0


def inference_single(args, model, img, dic_classid_classname=None, file=None,
                     save=False, images_crops=None, scores=None, x_norm=None, 
                     fp=None, masked_image=None):

    with torch.no_grad():
        outputs = model(img, ret_dist=True)

    if args.model_name == 'cal':
        outputs, images_crops = outputs
    if isinstance(outputs, tuple) and len(outputs) == 6:
        outputs, images_crops, x_norm, inter, scores_soft, scores = outputs
    elif isinstance(outputs, tuple) and len(outputs) == 5:
        outputs, x_norm, inter, scores_soft, scores = outputs
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        outputs, _ = outputs

    outputs = outputs.squeeze(0)
    for i, idx in enumerate(torch.topk(outputs, k=3).indices.tolist()):
        prob = torch.softmax(outputs, -1)[idx].item()
        if dic_classid_classname is not None:
            classname = dic_classid_classname[idx]
            out_text = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100)
            print(out_text)
        else:
            out_text = '[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100)
            print(out_text)
        if i == 0:
            top1_text = out_text

    if save:
        if args.model_name == 'cal':
            if args.cal_save_all:
                images_crops = rearrange(images_crops, 'b c h w k -> b c h (k w)')
            else:
                images_crops = images_crops[:, :, :, :, 0]

        fn = '{}.png'.format(os.path.splitext(os.path.split(file)[1])[0])
        fp = os.path.join(args.results_dir, fn)
        save_crops(img.detach().clone(), images_crops, fp, args.image_size,
                    args.pre_resize_factor, args.save_crops_only, args.custom_mean_std)

    if scores is not None and (args.vis_mask or args.vis_mask_all):
        masked_image = make_heatmaps(args, fp, img, inter, scores_soft,
                                        scores, x_norm, args.vis_mask_all, save=save)

    return top1_text, images_crops, masked_image


def inference_all(args):
    files_all = search_images(args)

    model, transform, dic_classid_classname = prepare_inference(args)

    for file in files_all:
        print(file)
        img = prepare_img(file, args, transform)

        # Classify
        inference_single(args, model, img, dic_classid_classname, file, save=True)

        if args.debugging:
            print('Finished.')
            return 0

    return 0


def main():
    args = parse_inference_args()
    inference_all(args)
    return 0


if __name__ == '__main__':
    main()

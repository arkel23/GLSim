import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def calc_avg_height_width(args):
    if 'ncfm' not in args.df_path:
        df = pd.read_csv(args.df_path)
        num_classes = len(np.unique(df['class_id'].to_numpy()))
        print(f'Dataframe {args.df_path} total number of images: {len(df)}, unique classes: {num_classes}')
    else:
        df = pd.read_csv(args.df_path, header=None, names=['dir'])

    w_list = []
    h_list = []

    for i, fn in enumerate(df['dir']):
        fp = os.path.join(os.path.split(args.df_path)[0], args.folder_images, fn)
        w, h = Image.open(fp).size
        w_list.append(w)
        h_list.append(h)
        if i % 10000 == 0:
            print(f'{i} / {len(df)}, file: {fp} width: {w} height: {h}')

    avg_width = sum(w_list) / len(df)
    avg_height = sum(h_list) / len(df)
    ratio = avg_width / avg_height
    # if avg_width > avg_height:
    #    ratio = avg_width / avg_height
    #else:
    #     ratio = avg_height / avg_width
    print(f'Average width: {avg_width} \t Average height: {avg_height} \t Ratio (larger to smaller): {ratio}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, help='path to data dic file', required=True)
    parser.add_argument('--folder_images', type=str, help='name of folder with images', required=True)
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.df_path))[0]
    print(args)

    calc_avg_height_width(args)


main()

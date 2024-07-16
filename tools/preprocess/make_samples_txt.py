import os
import yaml
import random
import argparse

import pandas as pd

DATASETS = ['cub', 'daf', 'dogs', 'flowers', 'food', 'inat17', 'moe', 'nabirds', 'pets', 'vegfru']
DATASETS_CATS = ['nabirds', 'daf', 'flowers', 'food', 'inat17', 'pets', 'vegfru']


def save_samples_ind(args):
    datasets = args.datasets_cats if args.datasets_cats else args.datasets
    num_crops = args.num_crops

    for dataset in datasets:
        dirs = []
        with open(f'configs/datasets/{dataset}.yaml') as f:
            cfg = yaml.safe_load(f)

        if dataset == 'daf':
            cfg['folder_test'] = 'fullMin256'

        df = pd.read_csv(f'data/{dataset}/test.csv')
        num_images = len(df)

        indexes = random.sample(range(num_images), num_crops)

        for idx in indexes:
            fp = os.path.join(cfg['dataset_root_path'], cfg['folder_test'], df.iloc[idx]['dir'])
            dirs.append(fp)

        df = pd.DataFrame(dirs, columns=['dir'])
        df.to_csv(f'samples_{dataset}.txt', header=True, index=False)

    return 0


def save_samples_all(args):
    datasets = args.datasets_cats if args.datasets_cats else args.datasets
    num_crops = args.num_crops

    dirs = []

    for dataset in datasets:
        with open(f'configs/datasets/{dataset}.yaml') as f:
            cfg = yaml.safe_load(f)
        df = pd.read_csv(f'data/{dataset}/test.csv')
        num_images = len(df)

        indexes = random.sample(range(num_images), num_crops)

        for idx in indexes:
            fp = os.path.join(cfg['dataset_root_path'], cfg['folder_test'], df.iloc[idx]['dir'])
            dirs.append(fp)
        # fp = os.path.join(cfg['dataset_root_path'], cfg['folder_test'], df.iloc[-1]['dir'])
        # dirs.append(fp)

    df = pd.DataFrame(dirs, columns=['dir'])
    df.to_csv('samples_all.txt', header=True, index=False)

    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_crops', type=int, default=10)
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    parser.add_argument('--datasets_cats', action='store_true',
                        help='if true then use DATASETS_CATS (7 instead of 10')
    parser.add_argument('--save_individually', action='store_false')

    args = parser.parse_args()

    if args.save_individually:
        save_samples_ind(args)
    else:
        save_samples_all(args)

if __name__ == '__main__':
    main()
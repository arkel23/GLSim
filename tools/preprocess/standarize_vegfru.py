import os
import argparse
import pandas as pd


def make_data_dic(args):
    for split in ('train', 'val', 'test'):
        fp = os.path.join(args.files_path, f'vegfru_{split}.txt')
        df = pd.read_csv(fp, sep=' ', header=None, names=['dir', 'class_id'])

        new = df['dir'].str.split('/', n=1, expand=True)
        df['dir'] = new[1]

        save_path = os.path.join(args.dataset_root_path, f'{split}.csv')
        df.to_csv(save_path, sep=',', header=True, index=False, columns=['class_id', 'dir'])
        print(f'Saved {fp} to {save_path}')

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_path', type=str, help='path to folder with text files')
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.files_path))[0]

    make_data_dic(args)


main()
import os
import argparse

import pandas as pd

from custom_datasets import NABirds, Aircraft, Cars, Dogs, Pets, Flowers, Food, INat17


DATASETS = ('cub', 'nabirds', 'aircraft', 'cars', 'dogs', 'pets', 'flowers',
            'food', 'inat17')


def make_classid_classname(args):
    if args.dataset_name == 'cub':
        args.dataset_root_path = os.path.split(os.path.normpath(args.classes_path))[0]
        # default_name: classes.txt
        df = pd.read_csv(args.classes_path, delimiter=r'\s+', names=['class_id', 'class_name'])
        df['class_id'] = df['class_id'] - 1

    elif args.dataset_name == 'nabirds':
        ds = NABirds(root=args.dataset_root_path, train=True, download=True)

        # labels in their tags are not continouous so use this label map to map it to continuous
        label_map = ds.label_map

        # access original label from pytorch continuous prediction label
        reverse_label_map = {v: k for k, v in label_map.items()}

        label_names = ds.class_names
        dic_classid_classname = {k: label_names[str(v)] for k, v in reverse_label_map.items()}

        df = pd.DataFrame.from_dict(dic_classid_classname, orient='index',
                                                      columns=['class_name'])
        df['class_id'] = df.index
        df = df[['class_id', 'class_name']]

        args.dataset_root_path = os.path.join(args.dataset_root_path, 'nabirds')

    elif args.dataset_name == 'inat17':
        ds = INat17(root=args.dataset_root_path, train=False)

        df = pd.DataFrame.from_dict(ds.dic_classid_classname, orient='index', columns=['class_name'])
        df['class_id'] = df.index

        df = df[['class_id', 'class_name']]
        df.sort_values(by='class_id', ascending=True, inplace=True)

    else:
        raise NotImplementedError

    save_fp = os.path.join(args.dataset_root_path, args.save_name)
    df.to_csv(save_fp, header=True, index=False)
    print(f'Saved df with length {len(df)} to {save_fp}: \n', df.head())

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=DATASETS, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str,
                        help='path to dataset root')
    parser.add_argument('--classes_path', type=str, help='path to filename with classes')
    parser.add_argument('--save_name', type=str, default='classid_classname.csv')
    args = parser.parse_args()

    make_classid_classname(args)


main()

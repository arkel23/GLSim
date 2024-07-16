import os
import argparse
import pandas as pd

from custom_datasets import CUB, NABirds, Aircraft, Cars, Dogs, Pets, Flowers, Food, INat17


DATASETS = ('cub', 'nabirds', 'aircraft', 'cars', 'dogs', 'pets', 'flowers',
            'food', 'inat17')


def make_df_from_dataset(args):

    if args.dataset_name == 'cub':
        base_folder = 'CUB_200_2011'
        dataset_train = CUB(root=args.dataset_root_path, train=True, download=True)
        dataset_test = CUB(root=args.dataset_root_path, train=False, download=True)
        print(len(dataset_train), dataset_train[0], len(dataset_train.class_names))
    elif args.dataset_name == 'nabirds':
        base_folder = 'nabirds'
        dataset_train = NABirds(root=args.dataset_root_path, train=True, download=True)
        dataset_test = NABirds(root=args.dataset_root_path, train=False, download=True)
    elif args.dataset_name == 'aircraft':
        base_folder = os.path.join('fgvc-aircraft-2013b', 'data')
        dataset_train = Aircraft(root=args.dataset_root_path, train=True, download=True)
        dataset_test = Aircraft(root=args.dataset_root_path, train=False, download=True)
    elif args.dataset_name == 'cars':
        base_folder = '.'
        dataset_train = Cars(root=args.dataset_root_path, train=True, download=True)
        dataset_test = Cars(root=args.dataset_root_path, train=False, download=True)
    elif args.dataset_name == 'dogs':
        base_folder = '.'
        dataset_train = Dogs(root=args.dataset_root_path, train=True, download=True)
        dataset_test = Dogs(root=args.dataset_root_path, train=False, download=True)
    elif args.dataset_name == 'pets':
        base_folder = 'oxford-iiit-pet'
        dataset_train = Pets(root=args.dataset_root_path, split='trainval', download=True)
        dataset_test = Pets(root=args.dataset_root_path, split='test', download=True)
    elif args.dataset_name == 'flowers':
        base_folder = 'flowers-102'
        if not args.val:
            dataset_train = Flowers(root=args.dataset_root_path, split='train', download=True)
        else:
            dataset_train = Flowers(root=args.dataset_root_path, split='val', download=True)
        dataset_test = Flowers(root=args.dataset_root_path, split='test', download=True)
    elif args.dataset_name == 'food':
        base_folder = 'food-101'
        dataset_train = Food(root=args.dataset_root_path, split='train', download=True)
        dataset_test = Food(root=args.dataset_root_path, split='test', download=True)
    elif args.dataset_name == 'inat17':
        base_folder = '.'
        dataset_train = INat17(root=args.dataset_root_path, train=True, download=True)
        dataset_test = INat17(root=args.dataset_root_path, train=False, download=True)
    else:
        raise NotImplementedError

    print('Test set:\n', len(dataset_test), dataset_test[0])

    for (split, dataset) in zip((args.save_name_train, args.save_name_test),
                                (dataset_train, dataset_test)):
        dic_target_img_dir = {}
        for i, (target, img_dir) in enumerate(dataset):
            dic_target_img_dir[i] = {'class_id': target, 'dir': img_dir}

        df = pd.DataFrame.from_dict(dic_target_img_dir, orient='index')
        print(f'Split: {split}\n', df.head(), len(df))

        fp = os.path.join(args.dataset_root_path, base_folder, f'{split}.csv')
        df.to_csv(fp, header=True, index=False)

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, required=True,
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=DATASETS, help='dataset name')
    parser.add_argument('--save_name_train', type=str, default='train_val')
    parser.add_argument('--save_name_test', type=str, default='test')
    parser.add_argument('--val', action='store_true', help='dl val set for flowers')
    args = parser.parse_args()
    make_df_from_dataset(args)


if __name__ == '__main__':
    main()

import os
import argparse
import pandas as pd


def data_split(args):
    # splits data into training and test

    df = pd.read_csv(args.df_path)
    print('Original df: ', len(df))

    n_per_class_df = df.groupby('class_id', as_index=True).count()

    df_list_train = []
    df_list_test = []
    for class_id, n_per_class in enumerate(n_per_class_df['dir']):
        train_samples_class = int(n_per_class*args.train_percent)
        test_samples_class = n_per_class - train_samples_class
        assert(train_samples_class+test_samples_class == n_per_class)
        train_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').head(train_samples_class)
        test_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').tail(test_samples_class)
        df_list_train.append(train_subset_class)
        df_list_test.append(test_subset_class)

    df_train = pd.concat(df_list_train)
    df_test = pd.concat(df_list_test)

    print('Train df: ')
    print(df_train.head())
    print(df_train.shape)
    print('test df: ')
    print(df_test.head())
    print(df_test.shape)

    df_name = f'{args.save_name_train}.csv'
    save_path = os.path.join(args.dataset_root_path, df_name)
    df_train.to_csv(save_path, sep=',', header=True, index=False)
    print(f'Saved {save_path} train dictionary.')

    df_name = f'{args.save_name_test}.csv'
    save_path = os.path.join(args.dataset_root_path, df_name)
    df_test.to_csv(save_path, sep=',', header=True, index=False)
    print(f'Saved {save_path} test dictionary')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, help='path to data dic file', required=True)
    parser.add_argument('--train_percent', type=float, default=0.8,
                        help='percent of data for training')
    parser.add_argument('--save_name_train', type=str, default='train')
    parser.add_argument('--save_name_test', type=str, default='val')
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.df_path))[0]
    print(args)

    data_split(args)


main()

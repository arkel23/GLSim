import argparse
import pandas as pd


def update_df(fn, og_new, count=False, sort=False):
    df = pd.read_csv(fn)
    print(df.head())

    df = df.assign(class_id=df.class_id.map(og_new))

    if sort:
        df = df.sort_values(by=['class_id'], ascending=True)

    print(df.head())
    df.to_csv(fn, sep=',', header=True, index=False)

    if count:
        print(df.groupby('class_id').count())


def main():
    '''Renumbers the classIDs to fit competition requirements (alphabetical)'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_files', type=str,
                        help='path to df with image paths')
    parser.add_argument('--df_id_name', type=str,
                        help='path to df with class id/names mapping')
    args = parser.parse_args()

    og_new = {0: 3, 1: 0, 2: 4, 3: 2, 4: 6, 5: 5, 6: 7, 7: 1}

    update_df(args.df_files, og_new, count=True)
    update_df(args.df_id_name, og_new, sort=True)


main()

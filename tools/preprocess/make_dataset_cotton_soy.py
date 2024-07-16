import os
import argparse
import glob
import pandas as pd
from PIL import Image


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(files_all))
    return files_all


def add_image_to_dics(i, fp, filename_classid_split_dic):
    # verify the image is RGB
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.mode == 'RGB':
        abs_path, filename = os.path.split(fp)
        class_id, _, split = os.path.splitext(filename)[0].split('_')
        class_id = int(class_id) - 1

        filename_classid_split_dic[i] = (filename, class_id, split)

    return 0


def save_filename_classid(args, filename_classid_split_dic):
    # save dataframe to hold the class IDs and the relative paths of the files
    df = pd.DataFrame.from_dict(
        filename_classid_split_dic, orient='index', columns=['dir', 'class_id', 'split'])
    print(df.head())

    no_classes = len(df['class_id'].unique())
    print('Total number of classes: ', no_classes)
    print('Total images files post-filtering (RGB only): ', len(filename_classid_split_dic))

    df.sort_values('class_id', inplace=True)
    print(df.head())

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] != 'train']
    print('Train set: ', len(df_train), len(df_train['class_id'].unique()))
    print('Test set: ', len(df_test),  len(df_test['class_id'].unique()))

    save_path_train = os.path.join(args.dataset_root_path, 'train_val.csv')
    df_train.to_csv(save_path_train, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    save_path_test = os.path.join(args.dataset_root_path, 'test.csv')
    df_test.to_csv(save_path_test, sep=',', header=True, index=False, columns=['class_id', 'dir'])
    return 0


def make_data_dic(args):
    '''
    makes an imagefolder (imagenet style) with images of class in a certain
    folder into a txt dictionary with the first column being the
    file dir (relative) and the second into the class
    '''
    files_all = search_images(args)

    # filename and classid pairs
    filename_classid_split_dic = {}

    for i, fp in enumerate(files_all):
        add_image_to_dics(i, fp, filename_classid_split_dic)

    # save filename_classid df
    save_filename_classid(args, filename_classid_split_dic)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='soybean200_square',
                        help='path to folder like IN')
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.images_path))[0]

    make_data_dic(args)


main()


import os
import argparse
import glob


def search_images(args):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files', len(files_all))
    return files_all


def make_data_dic(args):
    '''takes all images into folder and puts them into a file'''
    files_all = search_images(args)

    if args.append:
        f = open('test.csv', 'a')
    else:
        f = open('test.csv', 'w')

    for fn in files_all:
        if args.append:
            # when appending (test_stg2) add the whole relative path
            f.write('{}\n'.format(fn))
        else:
            # when writing (test_stg1) only write the filename
            f.write('{}\n'.format(os.path.basename(os.path.normpath(fn))))

    f.close()


def main():
    '''
    input is the path to the folder with test images
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to folder with test')
    parser.add_argument('--append', action='store_true', help='append to file')
    args = parser.parse_args()

    make_data_dic(args)


main()

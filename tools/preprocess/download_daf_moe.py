import argparse
import gdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces', action='store_true', help='dl dafre faces')
    parser.add_argument('--full', action='store_true', help='dl dafre full')
    parser.add_argument('--moe', action='store_true', help='dl moeimouto')

    args = parser.parse_args()

    if args.faces:
        gdown.download(id='184NpGg0wIYWj6KnOj3mN9psZ2r2JX-MZ',
                       output='./dafre_faces.tar.gz', quiet=False)
    if args.full:
        gdown.download(id='11mcQoIYsjk0N1AA-QftNJ6ngVKt69xia',
                       output='./dafre_full.tar.gz', quiet=False)
    if args.moe:
        gdown.download(id='1bEF1CrWLYfRJYauBY9bpiZ-LMnyEn24w',
                       output='./moeimouto_animefacecharacterdataset.tar.gz', quiet=False)


if __name__ == '__main__':
    main()

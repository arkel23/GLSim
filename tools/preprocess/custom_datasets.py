# https://github.com/pytorch/vision/tree/main/torchvision/datasets
# https://github.com/lvyilin/pytorch-fgvc-dataset
import os
import json
import pathlib
import warnings
from typing import Any, Callable, Optional, Union, Tuple, Sequence

import scipy.io
import numpy as np
import pandas as pd

# import torch
# import torch.utils.data as data
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, extract_archive, \
    download_and_extract_archive, verify_str_arg, check_integrity
from torchvision.datasets.utils import download_file_from_google_drive


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


class CUB(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CUB, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_dir = sample.filepath
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        return target, img_dir


class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
        super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
        if download:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        dataset_path = os.path.join(root, 'nabirds')
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.loader = default_loader
        self.train = train

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)
        self.image_class_labels = image_class_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_dir = sample.filepath
        target = self.label_map[sample.target]
        return target, img_dir


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    In the website for this dataset it says it has 102 classes and 10.2k images and so
    does everywhere else but the paper itself says 100 classes and opening it up
    contains only 100 classes and 10k images
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))
        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        return target, path

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def _download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (f'{image_ids[i]}.jpg', targets[i])
            # item = (os.path.join(self.root, self.img_folder,
            #                     '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    file_list = {
        'imgs': ('http://ai.stanford.edu/~jkrause/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://ai.stanford.edu/~jkrause/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = scipy.io.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.labels = []
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.labels.append(label)
                self.samples.append((path, label))\

        self.classes = np.unique(self.labels)

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.split(path)[1]
        return target, path

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

        self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_name, target = self._flat_breed_images[index]
        return target, image_name

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == \
                    len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))


class Pets(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:
                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.
            If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [f'{image_id}.jpg' for image_id in image_ids]
        # self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_dir = self._images[idx]
        target = self._labels[idx]
        return target, image_dir

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


class Flowers(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.
    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = pathlib.Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(f"image_{image_id:05d}.jpg")
            # self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.num_classes = len(set(self._labels))

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        return label, image_file

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


class Food(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.
    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            # self._image_files += [
            #    self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            # ]
            self._image_files += [f"{im_rel_path}.jpg" for im_rel_path in im_rel_paths]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        return label, image_file

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class INat17(VisionDataset):
    """`iNaturalist 17 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(INat17, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader

        split = 'train' if train else 'val'
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        else:
            raise RuntimeError('Needs to download previously using github links, dataset not found.')

        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']
        self.num_classes = len(set(self.annos[ind]['category_id'] for ind in range(self.__len__())))

        cats = all_annos['categories']
        list_ids = [cats[i]['id'] for i in range(len(cats))]
        list_names = [cats[i]['name'] for i in range(len(cats))]
        self.dic_classid_classname = {id_num: name for id_num, name in zip(list_ids, list_names)}

    def __getitem__(self, index):
        image_dir = self.images[index]['file_name']
        image_dir = image_dir.replace('train_val_images/', '')
        target = self.annos[index]['category_id']
        return target, image_dir

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))


if __name__ == '__main__':
    root = '/hdd/edwin/data/caltech_birds/original/'
    dataset = CUB(root=root, train=True, download=True)
    print(len(dataset), dataset[0], len(dataset.class_names))

    dataset = CUB(root=root, train=False, download=True)
    print(len(dataset), dataset[0])

    # must download manually, how many categories? 400, 555, 700, 1011 nodes?
    root = '/hdd/edwin/data/nabirds/'
    dataset = NABirds(root=root, train=True)
    print(len(dataset), dataset[0], len(dataset.class_names), len(dataset.class_hierarchy))

    dataset = NABirds(root=root, train=False)
    print(len(dataset), dataset[0])

    root = '/hdd/edwin/data/aircraft/'
    dataset = Aircraft(root=root, train=True, download=True)
    print(len(dataset), dataset[0], len(dataset.classes))

    dataset = Aircraft(root=root, train=False, download=True)
    print(len(dataset), dataset[0])

    root = '/hdd/edwin/data/cars/'
    dataset = Cars(root=root, train=True, download=True)
    print(len(dataset), dataset[0], len(dataset.classes))

    dataset = Cars(root=root, train=False, download=True)
    print(len(dataset), dataset[0])

    # Massive overlap with ImageNet. Could try to filter somehow or write about it.
    root = '/hdd/edwin/data/dogs/'
    dataset = Dogs(root=root, train=True, download=True)
    print(len(dataset), dataset[0], len(dataset._breeds))

    dataset = Dogs(root=root, train=False, download=True)
    print(len(dataset), dataset[0])

    root = '/hdd/edwin/data/pets/'
    dataset = Pets(root=root, split='trainval', download=True)
    print(len(dataset), dataset[0], len(dataset.classes))

    dataset = Pets(root=root, split='test', download=True)
    print(len(dataset), dataset[0])

    root = '/hdd/edwin/data/flowers/'
    dataset = Flowers(root=root, split='train', download=True)
    print(len(dataset), dataset[0], dataset.num_classes)

    dataset = Flowers(root=root, split='val', download=True)
    print(len(dataset), dataset[0])

    dataset = Flowers(root=root, split='test', download=True)
    print(len(dataset), dataset[0])

    root = '/hdd/edwin/data/food/'
    dataset = Food(root=root, split='train', download=True)
    print(len(dataset), dataset[0], len(dataset.classes))

    dataset = Food(root=root, split='test', download=True)
    print(len(dataset), dataset[0])

    '''
    root = '/hdd/edwin/data/inat17/'
    dataset = INat17(root=root, train=True)
    print(len(dataset), dataset[0])

    dataset = INat17(root=root, train=False)
    print(len(dataset), dataset[0])
    '''

# https://github.com/visipedia/inat_comp/tree/master/2017
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val2017.zip
unzip train_val2017.zip

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz
md5sum train_val_images.tar.gz
tar -xvzf train_val_images.tar.gz


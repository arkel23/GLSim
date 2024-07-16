nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name cub --dataset_root_path ../../data/cub
nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name cars --dataset_root_path ../../data/cars
nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name aircraft --dataset_root_path ../../data/aircraft
nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name dogs --dataset_root_path ../../data/dogs
nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name pets --dataset_root_path ../../data/pets
nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name food --dataset_root_path ../../data/food

# for flowers
#nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name flowers --dataset_root_path ../../data/flowers --save_name_train train
#nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name flowers --dataset_root_path ../../data/flowers --save_name_train val --val
#cd ../../data/flowers/flowers-102/
#cat train.csv > train_val.csv
# remove header, cat into train_val and then put header back
#nano val.csv
#cat val.csv >> train_val.csv
#nano val.csv

# need to download manually: nabirds, moe (anime), ncfm (fish), inat2017, vegfru, soy, cotton
# for nabirds:
#nohup python -u tools/preprocess/make_df_from_dataset.py --dataset_name nabirds --dataset_root_path ../../data/nabirds

# for ncfm:
# there's the ncfm_prepare_dataset.sh (basically download from kaggle and then can use provided df_*.csv files)

# for moeimouto:
# https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations?select=moeimouto-faces

# for inat17
# ./download_inat17.sh

# for vegfru:
# use official link (https://github.com/ustc-vim/vegfru which links to pan baidu) or kaggle (https://www.kaggle.com/datasets/zhaoyj688/vegfru/code)
# need to combine train and val similar to flowers
# need to move images from the veg and fru folders to one single folder called images
# mkdir images
# mv veg200_images/* fru92_images/* images/

# for soy and cotton:
# https://maxwell.ict.griffith.edu.au/cvipl/UFG_dataset.html

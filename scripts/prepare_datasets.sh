# cub
#python tools/preprocess/make_classid_classname.py --dataset_name cub --classes_path ../../data/cub/CUB_200_2011/classes.txt
#python tools/preprocess/data_split.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# moe
#python tools/preprocess/make_data_dic_imagenetstyle.py --images_path ../../data/moe/data/ --save_name moe.csv
#python tools/preprocess/data_split.py --df_path ../../data/moe/moe.csv --train_percent 0.8 --save_name_train train_val.csv --save_name_test test.csv
#python tools/preprocess/data_split.py --df_path ../../data/moe/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# dafb
#python tools/preprocess/data_split.py --df_path ../../data/daf/train_val.csv --train_percent 0.05 --save_name_train train.csv --save_name_test val.csv

# aircraft
python tools/preprocess/data_split.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# cars
python tools/preprocess/data_split.py --df_path ../../data/cars/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# food
python tools/preprocess/data_split.py --df_path ../../data/food/food-101/train_val.csv --train_percent 0.2 --save_name_train train.csv --save_name_test val.csv

# pets
python tools/preprocess/data_split.py --df_path ../../data/pets/oxford-iiit-pet/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# dogs
python tools/preprocess/data_split.py --df_path ../../data/dogs/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# nabirds
python tools/preprocess/data_split.py --df_path ../../data/nabirds/nabirds/train_val.csv --train_percent 0.8 --save_name_train train.csv --save_name_test val.csv

# flowers doesn't need to split as it has a default val split

# ncfm was already created in its own script

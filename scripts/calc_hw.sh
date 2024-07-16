# cub
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cub/CUB_200_2011/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cub/CUB_200_2011/test.csv --folder_images images

# nabirds
nohup python tools/preprocess/calc_hw.py --df_path ../../data/nabirds/nabirds/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/nabirds/nabirds/test.csv --folder_images images

# aircraft
nohup python tools/preprocess/calc_hw.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/aircraft/fgvc-aircraft-2013b/data/test.csv --folder_images images

# cars
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cars/train_val.csv --folder_images car_ims
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cars/test.csv --folder_images car_ims

# dogs
nohup python tools/preprocess/calc_hw.py --df_path ../../data/dogs/train_val.csv --folder_images Images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/dogs/test.csv --folder_images Images

# pets
nohup python tools/preprocess/calc_hw.py --df_path ../../data/pets/oxford-iiit-pet/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/pets/oxford-iiit-pet/test.csv --folder_images images

# flowers
nohup python tools/preprocess/calc_hw.py --df_path ../../data/flowers/flowers-102/train_val.csv --folder_images jpg
nohup python tools/preprocess/calc_hw.py --df_path ../../data/flowers/flowers-102/test.csv --folder_images jpg

# cotton
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cotton/train_val.csv --folder_images cotton_square_new
nohup python tools/preprocess/calc_hw.py --df_path ../../data/cotton/test.csv --folder_images cotton_square_new

# soy
nohup python tools/preprocess/calc_hw.py --df_path ../../data/soy/train_val.csv --folder_images soybean200_square
nohup python tools/preprocess/calc_hw.py --df_path ../../data/soy/test.csv --folder_images soybean200_square

# vegfru
nohup python tools/preprocess/calc_hw.py --df_path ../../data/vegfru/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/vegfru/test.csv --folder_images images

# inat17
#nohup python tools/preprocess/calc_hw.py --df_path ../../data/inat17/train_val.csv --folder_images train_val_images
#nohup python tools/preprocess/calc_hw.py --df_path ../../data/inat17/test.csv --folder_images train_val_images

# food
nohup python tools/preprocess/calc_hw.py --df_path ../../data/food/food-101/train_val.csv --folder_images images
nohup python tools/preprocess/calc_hw.py --df_path ../../data/food/food-101/test.csv --folder_images images

# ncfm fish
nohup python tools/preprocess/calc_hw.py --df_path ../../data/ncfm/train_val.csv --folder_images train
nohup python tools/preprocess/calc_hw.py --df_path ../../data/ncfm/test.csv --folder_images test

# moe
nohup python tools/preprocess/calc_hw.py --df_path ../../data/moe/train_val.csv --folder_images data
nohup python tools/preprocess/calc_hw.py --df_path ../../data/moe/test.csv --folder_images data

# dafb
nohup python tools/preprocess/calc_hw.py --df_path ../../data/daf/train_val.csv --folder_images fullMin256
nohup python tools/preprocess/calc_hw.py --df_path ../../data/daf/test.csv --folder_images fullMin256

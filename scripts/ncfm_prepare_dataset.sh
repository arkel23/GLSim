pip install kaggle
# put kaggle.json into ~/.kaggle/kaggle.json (if ubuntu) or C:\Users\<Windows-username>\.kaggle\kaggle.json (Windows)
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c the-nature-conservancy-fisheries-monitoring
unzip the-nature-conservancy-fisheries-monitoring.zip
unzip train.zip
unzip sample_submission_stg1.csv.zip
unzip sample_submission_stg2.csv.zip
unzip test_stg1.zip
sudo apt install p7zip-full p7zip-rar
7z x test_stg2.7z
rm *.zip *.7z

# take as input imagenet like dictionary and make a .csv with all files
python tools/preprocess/make_data_dic_imagenetstyle.py --images_path ../../data/ncfm/train/ --save_name fish.csv
# change classes ids to alphabetical order (required for submission)
python renumber_classes.py --df_files ../../data/fish.csv --df_id_name ../../data/classid_classname.csv
# split into 0.9 and 0.1 for train/val
python data_split.py --fn fish.csv
# create test.csv file for inference and submission
python prepare_test.py --path test_stg1
python prepare_test.py --path test_stg2 --append

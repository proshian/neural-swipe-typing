#!/bin/bash

python data_obtaining_and_preprocessing/download_yandex_cup_dataset_original.py

python data_obtaining_and_preprocessing/separate_grid.py \
    --input_dir ../data/data_original \
    --output_dir ../data/data_preprocessed

python data_obtaining_and_preprocessing/convert_validation_dataset_to_train_format.py \
    --dataset_path ../data/data_preprocessed/valid.jsonl \
    --ref_path ../data/data_original/valid.ref \
    --out_path ../data/data_preprocessed/valid.jsonl \
    --total 10000

cp ../data/data_original/voc.txt ../data/data_preprocessed/voc.txt

python data_obtaining_and_preprocessing/fix_grids.py \
    -i ../data/data_preprocessed/gridname_to_grid.json \
    -o ../data/data_preprocessed/gridname_to_grid__fixed.json

python -m data_obtaining_and_preprocessing.filter_dataset \
    --dataset_path ../data/data_preprocessed/train.jsonl \
    --grids_path ../data/data_preprocessed/gridname_to_grid.json \
    --output_path ../data/data_preprocessed/train_filtered.jsonl \
    --log_dir ../data/errors_in_original_data_swipes

rm ../data/data_preprocessed/train.jsonl

python -m data_obtaining_and_preprocessing.compute_trajectory_features_statistics \
    --train_data_path ../data/data_preprocessed/train_filtered.jsonl \
    --voc ../data/data_preprocessed/voc.txt \
    --output_json ../data/data_preprocessed/trajectory_features_statistics.json \
    --total 6000000

python -m data_obtaining_and_preprocessing.compute_key_bounding_box \
    --grids_path ../data/data_preprocessed/gridname_to_grid.json \
    --labels а б в г д е ë ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я \
    --output ../data/data_preprocessed/key_bounding_boxes.json

# NeuroSwipe

> [!Note]
> this branch is a snapshot of the project kept for reproducing the final Yandex Cup 2023 submission. It is not maintained; for the current state of the project see the [`main`](https://github.com/proshian/neural-swipe-typing/tree/main) branch.

Transformer model that is used for a gesture keyboard (performs recognition of curves that are swiped across a keyboard on a smartphone display)

The repository is my yandex cup 2023 solution (7-th place) with improvements

## Last Yandex Cup 2023 submission reproduction instructions

All commands are run from the repository root

0. Create and activate a virtual environment. Python 3.11 is recomended since the reproduction was tested with it. Since `torch==2.0.1` has no wheels for versions higher than 3.11 only versions 3.8-3.11 are supported. 

    To create and activate the environment you can use [uv](https://docs.astral.sh/uv/). This will also install Python 3.11.

    ```shell
    uv python install 3.11
    uv venv --seed --python 3.11 .venv
    source .venv/bin/activate
    ```

    Or, if Python 3.11 is already installed:

    ```shell
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

1. Install dependencies:

    ```shell
    python -m pip install -r sumbission_reproduction_requirements.txt
    ```
    
2. Download the dataset and unpack it into `./data/data` (e.g. the train set ends up at `./data/data/train.jsonl`):

    ```shell
    python ./src/downloaders/download_original_data.py
    ```

3. Get the official baseline predictions (they are used to fill in rows where the models produced fewer than 4 candidates):

    ```shell
    python ./src/keyboard_start/ks_lib/main.py --train-path data/data/train.jsonl --test-path data/data/test.jsonl --voc-path data/data/voc.txt --num-workers 4 --output-path ./results/submissions/baseline.csv
    ```

4. Convert the dataset so the `grid_name -> grid` mapping is stored in a separate `gridname_to_grid.json` file, and all dataset examples have `grid` replaced with `grid_name`. The converted dataset will appear in `./data/data_separated_grid`:

    ```shell
    python ./src/separate_grid.py
    cp ./data/data/voc.txt ./data/data_separated_grid/voc.txt
    ```

    Alternatively, download the output of `./src/separate_grid.py` from [Google Drive](https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej?usp=sharing) into `./data/data_separated_grid`:

    ```shell
    python ./src/downloaders/download_dataset_separated_grid.py
    ```

5. Download the weights for the final submission models from [Google Drive](https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6?usp=sharing) to `./results/final_submission_models/`:

    ```shell
    python ./src/downloaders/download_weights.py
    ```

6. Get the predictions of each model. They will end up in `./results/final_submission_predictions/test` as pickle files:

    ```shell
    python ./src/predict.py --num-workers 4 --config ./config-yandex-cup.json
    ```

7. Aggregate the predictions:

    ```shell
    python ./src/aggregate_predictions.py
    ```

    The final submission is written to `./results/submissions/my_last_submission.csv`.

## Demo

You can try out one of the models trained as part of the competition in a [web app](https://proshiang.pythonanywhere.com/)


![demo](https://github.com/proshian/yandex-cup-2023-ml-neuroswipe/assets/98213116/1c7210ab-f347-4f50-9105-c6eb74884827)


## Method

The model is encoder-decoder transformer.
The first tranformer encoder layer can input a sequence with elements of a dimension different from other encoder layers.

Encoder input sequence consists of elements denoted as `swipe point embedding` on the image below.

![Here should be an image of encoder_input_sequence_element](./REAME_materials/encoder_input_sequence_element.png)

The $\frac{dx}{dt}$, $\frac{dy}{dt}$, $\frac{d^2x}{dt^2}$, $\frac{d^2y}{dt^2}$ derivatives are calculated using finite difference method.

Decoder input sequence consists of character-level embeddings (with positional encoding) of the target word.

Keyboard key embeddings used in encoder and charracter embeddings used in decoder are different entities.

More info in [solution_description.md](solution_description.md) file (in Russian).

## Training
The training is in [src/train.ipynb notebook](src/train.ipynb)

<!-- Перед побучением необходимо очистить тренировочный датасет -->

## Future work

## For future me
See refactoring plan [here](./Refactoring_plan.md)

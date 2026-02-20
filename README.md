# Neural Swipe Typing

A transformer neural network for a gesture keyboard that transduces curves swiped across a keyboard into word candidates

Contribution:
* **A new method for constructing swipe point embeddings (SPE) that outperforms existing ones.** It leverages a weighted sum of all keyboard key embeddings, resulting in a notable perfomance boost: **0.67% increase in Swipe MRR and 0.73% in accuracy** compared to SPE construction methods described in literature

Other highlights:
* **Enhanced Inference with Custom Beam Search**: a modified beam search is implemented that masks out logits corresponding to impossible (according to dictionary) token continuations given an already generated prefix. It is faster and more accurate than a standard beam search

*This repository used to contain my Yandex Cup 2023 solution (7th place), but after many improvements, it has become a standalone project*

## Demo

Try out a live demo with a trained model from the competition through this [web app](https://proshian.pythonanywhere.com/)


![demo](./docs_and_assets/swipe_demos/demo.gif)

> [!Note]
> If the website is not available, you can run the demo yourself by following the instructions in [the web app's GitHub repository](https://github.com/proshian/neuroswipe_inference_web).

> [!Note]
> The website may take a minute to load, as it is not yet fully optimized. If you encounter a "Something went wrong" page, try refreshing the page. This usually resolves the issue.

> [!NOTE]  
> It is not guaranteed that the model used in the demo is up-to-date with the latest improvements in this repository. 

## Android Library

There is an [Android library](https://github.com/proshian/neural-swipe-keyboard-android) that aims to help to integrate models from this repository into android keyboards.

The library expects that the model is exported via ExecutorTorch (export script available at [`src/executorch_export.py`](src/executorch_export.py)).

> [!IMPORTANT]
> **Compatibility Notice**: The Android library is currently outdated. While the library includes an exported model that works with it, models exported from the current version of this training repository are **not compatible** with the library.
>
> If you need to export models for use with the Android library, use the [`executorch_investigation`](https://github.com/proshian/nst-claude/tree/executorch_investigation) branch of this repository, which is in a state compatible with the library.

## Report

**Access a brief research report [here](docs_and_assets/report/report.md)**, which includes:

* Overview of existing research
* Description of the developed method for constructing swipe point embeddings
* Comparative analysis and results

For in-depth insights, you can refer to my [master's thesis](https://drive.google.com/file/d/1ad9zlfgfy6kOA-41GxjUQIzr8cWuaqxL/view?usp=sharing) (in Russian)


## Setup

Set up a virtual environment and install dependencies:
```sh
pip3 install uv  
uv sync --all-extras
```

## Yandex Cup Dataset: Obtaining and Preparation

To acquire and prepare the Yandex Cup dataset, follow the steps below:

### Option 1: Obtain and Preprocess the Dataset from Scratch

```sh
cd src
bash ./data_obtaining_and_preprocessing/obtain_and_prepare_data.sh
```

> [!Note]  
> The pipeline takes approximately **6 hours** to complete on the tested machine.


### Option 2: Download the Preprocessed Dataset (Recommended)

If you prefer to skip the lengthy preprocessing steps, you can directly download the preprocessed dataset:

```sh
cd src
uv run ./data_obtaining_and_preprocessing/download_dataset_preprocessed.py
```



## Workflow Overview

Transducing swipes to a list of words involves multiple components

* SwipeFeatureExtractor instance
* neural network architecture
    * swipe point embedder
    * subword embedder
    * encoder
    * decoder
* model weights
* decoding algorithm


### SwipeFeatureExtractor
A `SwipeFeatureExtractor` is any callable that takes three integer 1d tensors (`x`, `y`, `t`) representing raw coordinates and time in milliseconds and returns a list of tensors that are inputs of a certain `SwipePointEmbedder`.
Current implementations of this protocol:
1. `TrajectoryFeatureExtractor`: Extracts trajectory features such as x, y, dt and coordinate derivatives.
2. `CoordinateFunctionFeatureExtractor`: An adapter to make callables that accept `torch.stack(x, y)` satisfy the `SwipeFeatureExtractor` interface. Example of these coordinate feature extractors:
    * `DistanceGetter` - for each swipe point returns the distance to the key centers
    * `NearestKeyGetter` - for each swipe point returns the id of the nearest key center
    * `KeyWeightsGetter` - for each swipe point returns the weights (importance) of the key by applying a function to the `DistanceGetter` output
3. `MultiFeatureExtractor`: Combines multiple feature extractors into one.


### Feature extraction in the dataset
`SwipeFeatureExtractor`s are used as a dataset transformation step to extract relevant features from the raw swipe data before feeding it into the model.

After collating the dataset, the format becomes `(packed_model_in, dec_out)`, where `packed_model_in` is `(encoder_input, decoder_input, swipe_pad_mask, word_pad_mask)`. `packed_model_in` is passed to the model via unpacking (`model(*packed_model_in)`).

* `encoder_input` is a list of tensors (padded features from a `SwipeFeatureExtractor`)
* `decoder_input` and `decoder_output` are `tokenized_target_word[1:]` and `tokenized_target_word[:-1]` correspondingly.


### Model
All current models are instances of `model.EncoderDecoderTransformerLike` and consist of the following components:
* Swipe point embedder
* Subword token embedder (currently char-level)
* Encoder
* Decoder


### WordGenerator

A WordGenerator receives the encoded swipe features for a swipe and outputs 
a sorted list of scored word candidates (list of tuples (word: str, score: float)).

A WordGenerator stores:
* A model (`EncoderDecoderTransformerLike`) that processes the encoded swipe features
* A subword_tokenizer (`CharLevelTokenizerv2`) that converts characters to tokens and vice versa
* A logit processor (`LogitProcessor`) that manipulates the model's output logits. Currently `VocabularyLogitProcessor` is used to apply vocabulary-based masking and make it impossible for the model to generate the tokens outside the vocabulary
* Hyperparameters specific to a particular word generator


Currently, word generators accept non batched swipe features (process one swipe at a time).

## Your Custom Dataset

The Dataset class expects a jsonl file with the following structure:

```json
[
    {
        "word":"на",
        "curve":{
            "x":[567,567,507, ...],
            "y":[66,66,101, ...],
            "t":[0,3,24, ...],
            "grid_name":"your_keyboard_layout_name"}
    },
    ...
]
```

You also need to add your keyboard layout to `grid_name_to_grid.json` and provide a tokenizer config (see the example in `tokenizers\keyboard\ru.json`)

You may want to clean the data from outliers and errors  using `src\data_obtaining_and_preprocessing\filter_dataset.py`

<!-- **TODO: This section needs more details.** -->

## Training


Train with the default config (uses configs/train.yaml):
```sh
uv run src/train.py
```

Train with a custom config:
```sh
uv run src/train.py --config-name experiment/traj_and_nearest_conformer
```

Override config values
```sh
uv run src/train.py --config-name experiment/traj_and_nearest_conformer encoder.params.dropout=0.2 decoder.params.dropout=0.2
```


You can also use as [train_for_kaggle.ipynb](src/train_for_kaggle.ipynb) jupyter notebook (for example if you want to do the training in kaggle).


## Prediction

[word_generation_demo.ipynb](src/word_generation_demo.ipynb) serves as an example on how to predict via a trained model.

[predict.py](src/predict.py) is used to obtain word candidates for a whole dataset and pickle them.

### Inheriting from train config (recommended)

When predicting with a trained model, inherit the architecture from the saved train config:

```sh
uv run src/predict.py --config-name predict_from_train \
  train_config_path="./lightning_logs/experiment_name/version_1/config.yaml" \
  model_weights_path="./model_states/model.pt" \
```

>[!Tip]
> Remember to escape all `=` signs in paths (train config path, model weights path, output path) with `\` to avoid parsing errors.

>[!Tip]
> `model_weights_path` can be either a `.ckpt` checkpoint file (includes more than just weights: optimizer state etc.) from lightning checkpoints or a `.pt` file with the model's state_dict. 


The output path is auto-derived: `results/predictions/{experiment_name}/{data_name}/{grid_name}/{checkpoint_name}.pkl`. Can be overridden by providing `output_path` in the config or via CLI.


Override any setting via CLI:

```sh
uv run src/predict.py --config-name predict_from_train \
  ... \
  generator.params.beamsize=10 \
  device=cpu
```


### Manual prediction config

You can also use self-contained config without inheriting from train config like the example `configs/predict_standalone_example.yaml`. Again, you can override any setting via CLI:

```sh
uv run src/predict.py \
  --config-name predict \
  model_weights_path=./model_states/model.pt \
  data_path=./data/test.jsonl \
  encoder=conformer
```


>[!Tip]
> On some systems you may find that multiprocessing with `num_workers > 0` is slower than `num_workers = 0`. Try both options to see which one works better for you.


### Predicting for all epochs

Any of the above methods can be used to predict for a single checkpoint. To predict for all checkpoints in a directory, use `predict_all_epochs.py`. It runs `predict.py` as a subprocess for each checkpoint found in the parent directory of `model_weights_path`, passing through all other arguments unchanged. The `model_weights_path` is required to be passed as a cli argument and is replaced with each checkpoint path (`*.pt` or `*.ckpt`) sequentially.

Example:

```sh 
uv run src/predict_all_epochs.py \
  --config-name predict_from_train \
  train_config_path="./lightning_logs/experiment_name/version_1/config.yaml" \
  model_weights_path="./checkpoints/experiment_name/epoch_end/model.ckpt"
```


## Evaluation

```sh
uv run -m src.evaluate --config configs/config_evaluation.json
```

## Plot metrics

Plot metrics from a CSV file obtained during evaluation (evaluate.py):
```sh
uv run -m src.plot_metrics --csv results/evaluation_results.csv --metrics accuracy mmr --output_dir results/plots --colors_config configs/experiment_colors.json
```

Plot metrics from TensorBoard logs obtained during training (train.py):
```sh
uv run -m src.plot_tb_metrics --tb_logdir_root lightning_logs --output_dir results/plots/tb --colors_config configs/experiment_colors.json
```

### Extracting model weights from a lightning checkpoint

You may want to extract the model weights from a lightning checkpoint (for example to share more lightweight .pt files instead of bulky .ckpt files with all the training state). Use the following script:

```sh
uv run -m src.utils.ckpt_to_pt --ckpt-path checkpoints --out-path model_states
```


## Yandex cup 2023 results
* [task](./docs_and_assets/yandex_cup/task/task.md)
* [submission reproduction](./docs_and_assets/yandex_cup/submission_reproduciton_instrucitons.md). 
* [leaderboard](./docs_and_assets/yandex_cup/leaderboard.md)


# Documentation
A WIP documentation can be found [here](./docs_and_assets/documentation.md). It doesn't contain much information yet, will be updated. Please refer to docstrings in the code for now


## Thank you for your attention
![thank_you](./docs_and_assets/swipe_demos/thank_you.gif)

## For future me
See [refactoring plan](./docs_and_assets/Refactoring_plan.md)

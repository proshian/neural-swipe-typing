# Neural glide typing

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
> The model is an old and underfit legacy transformer variation (m1_bigger in models.py) that was used in the competition. A significant update is planned for both this project and the web app, but it will happen in winter 2024 

## Report

**Access a brief research report [here](docs_and_assets/report/report.md)**, which includes:

* Overview of existing research
* Description of the developed method for constructing swipe point embeddings
* Comparative analysis and results

For in-depth insights, you can refer to my [master's thesis](https://drive.google.com/file/d/1ad9zlfgfy6kOA-41GxjUQIzr8cWuaqxL/view?usp=sharing) (in Russian)


## Prerequisites

Install the dependencies:

```sh
pip install -r requirements/requirements.txt
```

* The inference was tested with python 3.10
* The training was conducted in kaggle using Tesla P100



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
python ./data_obtaining_and_preprocessing/download_dataset_preprocessed.py
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

Your custom dataset must have items of format: `tuple(x, y, t, grid_name, tgt_word)`. These raw features won't be used but there are transforms defined in `feature_extractors.py` corresponding to every type of `swipe point embedding layer` that extract the needed features. You can apply these transforms in your dataset's `__init__` method or in `__get_item__` / `__iter__`. The data formats after transform and after collation are described above

You also need to add your keyboard layout to `grid_name_to_grid.json`

<!--

**TODO: Add info on how exactly the dataset should be integrated** 

-->

## Training

<!-- Перед побучением необходимо очистить тренировочный датасет -->

Use train.py with a train config. Example:
```sh
python -m src.train --train_config configs/train/train_traj_and_nearest.json
```

You can also use as [train_for_kaggle.ipynb](src/train_for_kaggle.ipynb) jupyter notebook (for example if you want to do the training in kaggle).


## Prediction

You may want to extract model states from checkpoints using the provided `ckpt_to_pt.py` script.
```sh
python -m src.utils.ckpt_to_pt --ckpt-path checkpoints --out-path model_states
```

[word_generation_demo.ipynb](src/word_generation_demo.ipynb) serves as an example on how to predict via a trained model.

[predict.py](src/predict.py) is used to obtain word candidates for a whole dataset and pickle them

predict.py usage example:

```sh
python src/predict.py --config configs/prediction/prediction_conf__traj_and_nearest.json --num-workers 4
```

```sh
python -m src.predict_all_epochs --config configs/prediction/prediction_conf__traj_and_nearest.json  --num-workers 4
```

## Evaluation

```sh
python -m src.evaluate --config configs/config_evaluation.json
```

## Plot metrics

```sh
python -m src.plot_metrics --csv results/evaluation_results.csv --metrics accuracy mmr --output_dir results/plots
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

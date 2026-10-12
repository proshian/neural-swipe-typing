from abc import ABC, abstractmethod
from typing import Tuple, List, Set, Dict, Callable, Iterable
import os
import pickle
import copy
import json

# Without this import pickle.load won't be able to load predictions.
from predict import Prediction


def remove_probs(dataset_preds: List[List[Tuple[float, str]]]
                 ) -> List[List[str]]:
    """
    Removes probabilities from a model predictions for a whole dataset.
    """
    new_preds = []
    for pred_line in dataset_preds:
        new_preds_line = []
        for _, word in pred_line:
            new_preds_line.append(word)
        new_preds.append(new_preds_line)
    return new_preds


def separate_out_vocab_single_crv(hypotheses_list: List[Tuple[float, str]],
                                  vocab_set: Set[str],
                                  ) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
    """
    Separates list of word hypotheses for a single curve into a
    list of in_vocab words and a list of out_vocab words.

    Arguments:
    ----------
    hypotheses_list: List[Tuple[float, str]]
        List of word hypotheses for a single curve. Each element is a tuple
        (score: float, word: str) where score = log(prob(word)).
    vocab_set: Set[str]
        Set of in_vocab words.
    """
    in_vocab_words = []
    out_vocab_words = []

    for score, word in hypotheses_list:
        if word in vocab_set:
            in_vocab_words.append((score, word))
        else:
            out_vocab_words.append((score, word))
    # if limit != None:
    #     in_vocab_words = in_vocab_words[:limit]
    return in_vocab_words, out_vocab_words


def separate_out_vocab_all_crvs(dataset_preds: List[List[str]],
                                vocab_set: Set[str]
                                ) -> Tuple[List[List[str]], Dict[int, List[str]]]:
    """
    Separates model predictions for a whole dataset into two lists:
    1) list of lists of in_vocab words for each curve
    2) dict of lists of out_vocab words for each curve. If all hypotheses for
        a curve are in_vocab, then the curve_idx is not present in the dict.
    """
    all_in_vocab_preds = []
    all_errorous_word_preds = {}

    for i, hypotheses_lst in enumerate(dataset_preds):
        in_vocab_words, out_vocab_words = separate_out_vocab_single_crv(
            hypotheses_lst, vocab_set)

        all_in_vocab_preds.append(in_vocab_words)
        all_errorous_word_preds[i] = out_vocab_words

    return all_in_vocab_preds, all_errorous_word_preds


def append_preds(original_preds: List[List[str]],
                 additional_preds: List[List[str]],
                 limit: int) -> List[List[str]]:
    """
    Creates a new list of predictions by appending words
    from additional_preds to a copy of original_preds, skipping
    the words that are already present in original_preds.
    """
    merged_preds = copy.deepcopy(original_preds)

    for original_line, additional_line in zip(merged_preds, additional_preds):
        for additional_el in additional_line:
            if len(original_line) >= limit:
                break
            if additional_el not in original_line:
                original_line.append(additional_el)

    return merged_preds


def merge_default_and_extra_preds(
        default_preds: List[List[str]],
        extra_preds: List[List[str]],
        default_idxs: List[int],
        extra_idxs: List[int]) -> List[List[str]]:
    """
    Merges predictions for default and extra grid subsets of the dataset
    into list of predictions for the whole dataset.
    """
    preds = [None] * (len(default_preds) + len(extra_preds))

    for i, val in zip(default_idxs, default_preds):
        preds[i] = copy.deepcopy(val)
    for i, val in zip(extra_idxs, extra_preds):
        preds[i] = copy.deepcopy(val)

    return preds


def get_vocab_set(vocab_path: str):
    with open(vocab_path, 'r', encoding = "utf-8") as f:
        return set(f.read().splitlines())
    
    
def load_preds_to_aggregate(paths: List[str]
                            ) -> List[List[List[Tuple[float, str]]]]:
    preds_to_aggregate = []
    for f_path in paths:
        with open(f_path, 'rb') as f:
            prediction_and_meta = pickle.load(f)
        prediction = prediction_and_meta.prediction
        preds_to_aggregate.append(prediction)
    return preds_to_aggregate


def load_baseline_preds(path: str) -> List[List[str]]:
    baseline_preds = None
    with open(path, 'r', encoding = 'utf-8') as f:
        baseline_preds = f.read().splitlines()
    baseline_preds = [line.split(",") for line in baseline_preds]
    return baseline_preds
    

def get_default_and_extra_idxs(dataset_path) -> Tuple[List[int], List[int]]:
    """
    Gets indices of the dataset examples of default and extra grids.

    Arguments:
    ----------
    dataset_path: str

    Returns:
    --------
    default_idxs: List[int]
        List of indices of the dataset examples of default grid.
    extra_idxs: List[int]
        List of indices of the dataset examples of extra grid.
    """
    default_idxs = []
    extra_idx = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_data = json.loads(line)
            grid_name = line_data['curve']['grid_name']
            if grid_name == 'default':
                default_idxs.append(i)
            elif grid_name == 'extra':
                extra_idx.append(i)
            else:
                raise ValueError(f"Unexpected grid_name: {grid_name}.")
    
    return default_idxs, extra_idx


def create_submission(preds_list: List[List[str]],
                      out_path: str):
    """
    Arguments:
    ----------
    preds_list: List[List[str]]
        List of predictions for the whole dataset. Each row is a list of
        0 to `max_n_hypothesis` words.
    out_path: str
        Path to the output file.
    """
    if os.path.exists(out_path):
        raise ValueError(f"File {out_path} already exists")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for preds in preds_list:
            pred_str = ",".join(preds)
            f.write(pred_str + "\n")


def aggregate_preds_processed_appendage(preds_to_aggregate: List[List[List[str]]],
                                        limit: int) -> List[List[str]]:
    """
    Aggregates all predictions by appending words from each model
    in a given order to the resulting whole_dataset_predictions,
    avoiding duplicates for same curve.

    Arguments:
    ----------
    preds_to_aggregate: List[List[List[str]]]
        List of whole_dataset_predictions for each model. Should
        be processed (no probs, no out_vocab words).
    limit: int
        Maximum number of hypothesis per curve to output.
    """
    dataset_len = len(preds_to_aggregate[0])
    aggregated_preds = [[] for _ in range(dataset_len)]

    while preds_to_aggregate:
        aggregated_preds = append_preds(aggregated_preds,
                                        preds_to_aggregate.pop(0),
                                        limit = limit)    
    return aggregated_preds


def aggregate_preds_raw_appendage(raw_preds_list: List[List[List[str]]],
                                  vocab_set: Set[str],
                                  limit: int) -> List[List[str]]:
    """
    Prepares raw_preds_list for aggregation and calls aggregate_preds_processed_appendage.
    """
    preds_to_aggregate = [separate_out_vocab_all_crvs(preds, vocab_set)[0]
                          for preds in raw_preds_list]
    preds_to_aggregate = [remove_probs(preds) for preds in preds_to_aggregate]
    return aggregate_preds_processed_appendage(preds_to_aggregate, limit = limit)


def scale_probs(preds: List[List[Tuple[float, str]]],
                weight: float
                ) -> List[List[Tuple[float, str]]]:
    """
    Multiplies all probabilities in a models predictions by a given weight.
    
    Arguments:
    ----------
    preds: List[List[Tuple[float, str]]]
        List of word hypotheses for a whole dataset of a single model.
        Each element is a list of tuples (score: float, word: str)
        where score = log(prob(word)).
    weight: List[float]
        Weight to scale probabilities.
    """
    scaled_preds = []
    for preds_line in preds:
        scaled_preds_line = []
        for score, word in preds_line:
            scaled_preds_line.append((score * weight, word))
        scaled_preds.append(scaled_preds_line)
    return scaled_preds



def merge_sorted_lists(l1: list, l2: list, key: Callable) -> list:
    """
    Merges two sorted lists into one sorted list.
    """
    merged = []
    i1 = 0
    i2 = 0
    while i1 < len(l1) and i2 < len(l2):
        if key(l1[i1]) < key(l2[i2]):
            merged.append(l1[i1])
            i1 += 1
        else:
            merged.append(l2[i2])
            i2 += 1
    if i1 < len(l1):
        merged.extend(l1[i1:])
    if i2 < len(l2):
        merged.extend(l2[i2:])

    return merged


def merge_sorted_preds(model_preds_list: List[List[List[Tuple[float, str]]]]
                       ) -> List[List[Tuple[float, str]]]:
    """
    Merges predictions for a whole dataset from several models into one
    list of predictions for a whole dataset. The resulting list is sorted
    by score. It's supposed that each model prediction is sorted by score.
    """
    dataset_len = len(model_preds_list[0])
    merged_preds = []
    for i in range(dataset_len):
        merged_preds_line = []
        for model_preds in model_preds_list:
            merged_preds_line = merge_sorted_lists(merged_preds_line,
                                                   model_preds[i],
                                                   key = lambda x: x[0])
        merged_preds.append(merged_preds_line)
    return merged_preds


def delete_duplicates_stable(lst: Iterable) -> list:
    """
    Deletes duplicates from a list. Maintains the order of elements.
    """
    new_lst = []
    new_lst_set = set()
    for el in lst:
        if el not in new_lst_set:
            new_lst.append(el)
            new_lst_set.add(el)
    return new_lst


def aggregate_preds_raw_weighted(raw_preds_list: List[List[List[str]]],
                                 weights: List[float],
                                 vocab_set: Set[str],
                                 limit: int) -> List[List[str]]:
    """
    Aggregates predictions by multiplying each probability in a models
    predictions by a corrisponding weight and than merging and sorting
    all rows.

    Arguments:
    ----------
    preds_to_aggregate: List[List[List[str]]]
        List of whole_dataset_predictions for each model. Should
        be processed (no out_vocab words).
    weights: List[float]
        List of weights for each model.
    vocab_set: Set[str]
    limit: int
        Maximum number of hypothesis per curve to output.
    """
    preds_to_aggregate = [separate_out_vocab_all_crvs(preds, vocab_set)[0]
                          for preds in raw_preds_list]
    preds_to_aggregate = [scale_probs(preds, weight)
                          for preds, weight
                          in zip(preds_to_aggregate, weights)]
    preds = merge_sorted_preds(preds_to_aggregate)
    preds = remove_probs(preds)
    preds = [delete_duplicates_stable(preds_line) for preds_line in preds]
    preds = [preds_line[:limit] for preds_line in preds]

    return preds


# Aggregators are basicly functions. They are impplemented
# as callable classes to define an interface
class PredictionsAgregator(ABC):
    # there may be a separate method that aggregates
    # predictions into same format (tuples(score, word))

    @abstractmethod
    def __call__(raw_preds_list: List[List[List[Tuple[float, str]]]],
                 max_n_hypothesis: int = 4,
                 ) -> List[List[str]]:
        """
        Aggregates a list of predictions into one prediciton.

        Arguments:
        ----------
        raw_preds_list: List[List[List[str]]]
            List of model_predictions_list several models predictions. 
            Each models prediction has `dataset_len` rows. Each row
            is a list with tuple elements: (score: float, word: str)
            where score = log(prob(word)). A row may be an empty list.
        max_n_hypothesis: int
            Maximum number of hypothesis per curve to output.
        
        Returns:
        agregated_predictions: List[List[Tuple[float, str]]]
            agregated_predictions[i][j] is j-th word hypothesis
            for i-th curve in the dataset. Each row consists of 0 to 
            `max_n_hypothesis` words.
        """
        pass


class AppendAgregator(PredictionsAgregator):
    def __call__(raw_preds_list: List[List[List[Tuple[float, str]]]],
                 max_n_hypothesis: int = 4,
                 ) -> List[List[Tuple[float, str]]]:
        """
        Aggregates all predictions by appending words from each model
        in a given order to the resulting whole_dataset_predictions,
        avoiding duplicates for same curve.
        It's supposed that raw_preds_list is sorted by
        models MMR score on validation.
        """
        pass


class WeightedAgregator(PredictionsAgregator):
    def __call__(raw_preds_list: List[List[List[Tuple[float, str]]]],
                 max_n_hypothesis: int = 4,
                 ) -> List[List[Tuple[float, str]]]:
        """
        Aggregates predictions by multiplying each probability in a models
        predictions by a corrisponding weight and than merging and sorting
        all rows.
        """
        pass



if __name__ == "__main__":
    DATA_ROOT = "data/data_separated_grid/"

    grid_name_to_ranged_preds_names = {
        'default': [
            "m1_bigger__m1_bigger_v2__2023_11_12__14_51_49__0.13115__greed_acc_0.86034__default_l2_0_ls0_switch_2.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__12_30_29__0.13121__greed_acc_0.86098__default_l2_0_ls0_switch_2.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_11__22_18_35__0.13542_default_l2_0_ls0_switch_1.pt.pkl",
            "m1_v2__m1_v2__2023_11_09__10_36_02__0.14229_default_switch_0.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__00_39_33__0.13297_default_l2_0_ls0_switch_1.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_11__14_29_37__0.13679_default_l2_0_ls0_switch_0.pt.pkl",
            
        ],
        'extra': [
            "m1_v2__m1_v2__2023_11_09__17_47_40__0.14301_extra_l2_1e-05_switch_0.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__02_27_14__0.13413_extra_l2_0_ls0_switch_1.pt.pkl"
        ]
    }

    vocab_set = get_vocab_set(os.path.join(DATA_ROOT, "voc.txt"))

    default_idxs, extra_idxs = get_default_and_extra_idxs(
        os.path.join(DATA_ROOT, "test.jsonl"))

    grid_name_to_aggregated_preds = {}

    for grid_name in ('default', 'extra'):
        f_names = grid_name_to_ranged_preds_names[grid_name]
        f_paths = [os.path.join("results/final_submission_predictions/test/", f_name)
                   for f_name in f_names]
        
        preds_to_aggregate = load_preds_to_aggregate(f_paths)
        
        aggregated_preds = aggregate_preds_raw_appendage(
            preds_to_aggregate,
            vocab_set,
            limit = 4)

        grid_name_to_aggregated_preds[grid_name] = aggregated_preds
        

    full_preds = merge_default_and_extra_preds(
        grid_name_to_aggregated_preds['default'],
        grid_name_to_aggregated_preds['extra'],
        default_idxs,
        extra_idxs)
    

    baseline_preds = load_baseline_preds("results/submissions/baseline.csv")
    full_preds = append_preds(full_preds, baseline_preds, limit = 4)

    create_submission(full_preds,
        f"results/submissions/my_last_submission.csv")
    
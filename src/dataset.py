import json
from collections.abc import Callable
from typing import Optional, List, Tuple, Dict
import array

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from feature_extraction.swipe_feature_extractors import SwipeFeatureExtractor
from ns_tokenizers import CharLevelTokenizerv2
from torch_serialized_list import TorchSerializedList


RawDatasetEl = Tuple[array.array, array.array, 
                     array.array, str, Optional[str]]


def _get_data_from_json_line(line) -> RawDatasetEl:
    data = json.loads(line)

    X = array.array('h', data['curve']['x'])
    Y = array.array('h', data['curve']['y'])
    T = array.array('h', data['curve']['t'])

    grid_name = data['curve']['grid_name']   

    tgt_word = data['word'] if 'word' in data else None

    return X, Y, T, grid_name, tgt_word


class SwipeDataset(Dataset):
    """
    Dataset class for NeuroSwipe jsonl dataset
    """

    def __init__(self,
                 data_path: str,
                 word_tokenizer: CharLevelTokenizerv2,
                 grid_name_to_swipe_feature_extractor: Dict[str, SwipeFeatureExtractor],
                 use_serialized_list: bool = True,
                 total: Optional[int] = None):
        """
        Arguments:
        ----------
        data_path: str
            Path to the NeuroSwipe dataset in JSON format.
            A custom version of the dataset is used: "grid" property
            is replaced with "grid_name". The grid itself is stored in
            a separate gridname_to_grid dictionary.
            Dataset is a list of JSON lines. Each line is a dictionary
            with the following properties:
            - word (str): word that was typed. 
                Is abscent in test and val datasets.
            - curve (dict): dictionary that contains the following properties:
                - x (List[int]): x coordinates of the swipe trajectory.
                - y (List[int]): y coordinates of the swipe trajectory.
                - t (List[int]): time (in ms) from the beginning of the swipe.
                - grid_name (str): name of the keyboard grid.
        total: Optional[int]
            Number of dataset elements. Is used only for progress bar.
        """
        self.use_serialized_list = use_serialized_list
        self.data_list = self._get_data(
            data_path, total=total)
        if use_serialized_list:
            self.data_list = TorchSerializedList(
                self.data_list
            )
        self.word_tokenizer = word_tokenizer
        self.grid_name_to_swipe_feature_extractor = grid_name_to_swipe_feature_extractor
        
    def _get_data(self,
                  data_path: str,
                  transform: Optional[Callable] = None,
                  total: Optional[int] = None) -> List[RawDatasetEl]:
        data_list = []
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total = total):
                data_el = self._get_data_from_json_line(line)
                if transform is not None:
                    data_el = transform(data_el)
                data_list.append(data_el)
        return data_list

    def _get_data_from_json_line(self,
                                 line
                                 ) -> RawDatasetEl:
        return _get_data_from_json_line(line)
    
    def _get_decoder_in_and_out(self, tgt_word: str
                                ) -> Tuple[Tensor, Tensor]:
        tgt_token_seq: List[int] = self.word_tokenizer.encode(tgt_word)
        tgt_token_seq = torch.tensor(tgt_token_seq, dtype=torch.int64)
        decoder_in = tgt_token_seq[:-1]
        decoder_out = tgt_token_seq[1:]
        return decoder_in, decoder_out
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int
                    ) -> Tuple[Tuple[List[Tensor], Tensor], Tensor]:
        x, y, t, grid_name, tgt_word = self.data_list[idx]
        x, y, t = map(
            lambda x: torch.tensor(x, dtype=torch.float32), 
            (x, y, t))
        swipe_feature_extractor = self.grid_name_to_swipe_feature_extractor[grid_name]
        swipe_features = swipe_feature_extractor(x, y, t)
        decoder_in, decoder_out = self._get_decoder_in_and_out(tgt_word)
        
        return ((swipe_features, decoder_in), decoder_out)
    
    @classmethod
    def from_data_list(cls, 
                       data_list: list, 
                       word_tokenizer: CharLevelTokenizerv2,
                       grid_name_to_swipe_feature_extractor: Dict[str, SwipeFeatureExtractor],
                       ):        
        obj = cls.__new__(cls)

        obj.data_list = data_list
        obj.grid_name_to_swipe_feature_extractor = grid_name_to_swipe_feature_extractor
        obj.word_tokenizer = word_tokenizer

        return obj



class SwipeDatasetSubset:
    def __init__(self, dataset: SwipeDataset, grid_name: str):
        self.grid_name_list = [el[3] for el in dataset.data_list]
        self.dataset = dataset
        self.grid_name = grid_name
        self.grid_idxs = self._get_grid_idxs()
    
    def _get_grid_idxs(self):
        return [i for i, gname in enumerate(self.grid_name_list)
                if gname == self.grid_name]
    
    def __len__(self):
        return len(self.grid_idxs)
    
    def __getitem__(self, idx):
        return self.dataset[self.grid_idxs[idx]]



class CollateFn:
    def __init__(self, batch_first: bool, word_pad_idx: int, 
                 swipe_pad_idx: int = 0) -> None:
        self.word_pad_idx = word_pad_idx
        self.batch_first = batch_first
        self.swipe_pad_idx = swipe_pad_idx


    def __call__(self, batch: list):
        """
        Given a List where each row is 
        ((encoder_in_sample, decoder_in_sample), decoder_out_sample) 
        returns a tuple of two elements:
        1. (encoder_in, decoder_in, swipe_pad_mask, word_pad_mask)
        2. decoder_out

        Arguments:
        ----------
        batch: list of tuples:
            ((encoder_in, dec_in_char_seq), dec_out_char_seq),
            where encoder_in is a list of torch tensors

        Returns:
        --------
        1. transformer_in: tuple of torch tensors:
            (enc_in, dec_in, swipe_pad_mask, word_pad_mask),
            where enc_in can be either a single tensor or a tuple
            of two tensors (depends on type of input)
            Each element is a torch tensor of shape:
            - enc_in: list of tensors with shapes:
                [(swipe_len, batch_size, n_feats1), (swipe_len, batch_size, n_feats2), ...]
            - dec_in: (chars_seq_len - 1, batch_size)
            - swipe_pad_mask: (batch_size, swipe_len)
            - word_pad_mask: (batch_size, chars_seq_len - 1)
        2. dec_out: torch tensor of shape (chars_seq_len - 1, batch_size)
        """
        decoder_inputs, decoder_outputs = [], []

        num_encoder_features = len(batch[0][0][0])
        encoder_inputs = [[] for _ in range(num_encoder_features)]

        for (enc_in, dec_in), dec_out in batch:
            for feature, features_list in zip(enc_in, encoder_inputs):
                features_list.append(feature)

            decoder_inputs.append(dec_in)
            decoder_outputs.append(dec_out)

        encoder_inputs_padded = [
            pad_sequence(
                encoder_in_el, batch_first=self.batch_first,
                padding_value=self.swipe_pad_idx)
            for encoder_in_el in encoder_inputs]

        decoder_inputs_padded = pad_sequence(
            decoder_inputs, batch_first=self.batch_first,
            padding_value=self.word_pad_idx)
        
        decoder_outputs_padded = pad_sequence(
            decoder_outputs, batch_first=self.batch_first,
            padding_value=self.word_pad_idx)
        

        word_pad_mask = decoder_inputs_padded == self.word_pad_idx
        if not self.batch_first:
            word_pad_mask = word_pad_mask.T  # word_pad_mask is always batch first


        encoder_in_el = encoder_inputs_padded[0]
        max_swipe_len = encoder_in_el.shape[1] if self.batch_first else encoder_in_el.shape[0]
        encoder_inputs_single_feature_no_pad = encoder_inputs[0]
        encoder_input_lengths = torch.tensor(
            [len(x) for x in encoder_inputs_single_feature_no_pad])

        batch_size = encoder_input_lengths.shape[0]

        # Create mask where True indicates positions beyond a 
        # corresponding trajectory length.
        # 1. Create index sequences [0,1,…max_swipe_len-1] 
        #    for each batch element.
        # 2. Compare each index of each batch element 
        #    with the length of the corresponding trajectory.
        # Shape: (batch_size, max_swipe_len)
        swipe_pad_mask = torch.arange(max_swipe_len).expand(
            batch_size, max_swipe_len) >= encoder_input_lengths.unsqueeze(1)
        

        transformer_in = (encoder_inputs_padded, 
                          decoder_inputs_padded, 
                          swipe_pad_mask, 
                          word_pad_mask)
        
        return transformer_in, decoder_outputs_padded

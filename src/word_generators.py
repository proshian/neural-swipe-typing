from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import heapq

import torch
import torch.nn.functional as F
from torch import Tensor

from ns_tokenizers import CharLevelTokenizerv2
from model import EncoderDecoderTransformerLike
from logit_processors import LogitProcessor


def _prepare_encoder_input(encoder_in: list[Tensor], 
                           device: str, batch_first: bool
                           ) -> Tuple[Tensor, Tensor]:
    encoder_in = [el.unsqueeze(0 if batch_first else 1) for el in encoder_in]
    encoder_in = [el.to(device) for el in encoder_in]
    return encoder_in


def move_encoder_in_to_device(encoder_in: list[Tensor], 
                              device: str) -> list[Tensor]:
    return [el.to(device) for el in encoder_in]


class WordGenerator(ABC):
    def __init__(
        self,
        model: EncoderDecoderTransformerLike,
        tokenizer: CharLevelTokenizerv2,
        device,
        logit_processor: Optional[LogitProcessor] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = tokenizer.char_to_idx['<eos>']
        self.logit_processor = logit_processor

    def switch_model(self, model: EncoderDecoderTransformerLike):
        self.model = model

    @abstractmethod
    def __call__(self, swipe_features: List[Tensor]) -> List[Tuple[float, str]]:
        pass



class GreedyGenerator(WordGenerator):
    def __init__(
        self,
        model: EncoderDecoderTransformerLike,
        tokenizer: CharLevelTokenizerv2,
        device,
        logit_processor: Optional[LogitProcessor] = None,
        max_steps_n: int = 35,
    ):
        super().__init__(model, tokenizer, device, logit_processor)
        self.max_steps_n = max_steps_n

    @torch.inference_mode()
    def _generate(self, swipe_features: List[Tensor]) -> List[Tuple[float, str]]:
        BATCH_SIZE_DIM = 1
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        log_prob = 0.0

        swipe_features = _prepare_encoder_input(swipe_features, self.device, False)
        encoded = self.model.encode(swipe_features, None)

        for _ in range(self.max_steps_n):
            dec_in_char_seq = torch.tensor(tokens, device=self.device).unsqueeze_(BATCH_SIZE_DIM)
            next_tokens_logits: Tensor = self.model.decode(
                dec_in_char_seq, encoded, None, None).squeeze_(BATCH_SIZE_DIM)[-1]
            if self.logit_processor:
                next_tokens_logits = self.logit_processor.process(
                    next_tokens_logits, tokens)
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1)
            best_next_token = int(next_tokens_logproba.argmax())
            log_prob += float(next_tokens_logproba[best_next_token])
            
            tokens.append(best_next_token)
            if best_next_token == self.eos_token_id:
                break

        return [(-log_prob, self.tokenizer.decode(tokens[1:-1]))]

    def __call__(self, encoder_in) -> List[Tuple[float, str]]:
        return self._generate(encoder_in)
    
    def generate_word_only(self, encoder_in) -> str:
        return self._generate(encoder_in)[0][1]




class BeamGenerator(WordGenerator):
    def __init__(
        self,
        model: EncoderDecoderTransformerLike,
        tokenizer: CharLevelTokenizerv2,
        device,
        logit_processor: Optional[LogitProcessor] = None,
        max_steps_n: int = 35,
        beamsize: int = 6,
        normalization_factor: float = 0.5,
        return_hypotheses_n: Optional[int] = None,
    ) -> None:
        """
        Arguments:
        ----------
        return_hypotheses_n: Optional[int]
            return_hypotheses_n: Number of best hypotheses to return. If None,
            returns all found hypotheses.
        """
        super().__init__(model, tokenizer, device, logit_processor)
        self.max_steps_n = max_steps_n
        self.beamsize = beamsize
        self.normalization_factor = normalization_factor
        self.return_hypotheses_n = return_hypotheses_n

    @torch.inference_mode()
    def __call__(self,
                 encoder_in) -> List[Tuple[float, str]]:
        """
        Returns:
        --------
        List of tuples (score, text), where:
        - score is the normalized **negative** log probability of the hypothesis
        - text is the decoded word
        The list is sorted by score (best hypotheses first) and contains
        min(return_hypotheses_n, total_hypotheses_found) items if
        return_hypotheses_n is specified, or all found hypotheses otherwise.
        """
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        initial_length = len(tokens)

        # Partial hypotheses is a heap (stored as a list) of tuples.
        # Each tuple consists of a partial (unfinished aka intermidiate)
        # hypothesis and it's weight.
        # Weight is a measure of likelihood of the hypothesis.
        # [(w1, hypothesis1), (w2, hypothesis2), ...] 
        partial_hypotheses = [(0, tokens)]
        final_hypotheses = []


        encoder_in = _prepare_encoder_input(encoder_in, self.device, False)

        encoded = self.model.encode(encoder_in, None)

        while len(partial_hypotheses) > 0:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)


            dec_in_char_seq = torch.tensor(cur_partial_hypothesis, device=self.device).reshape(-1, 1)  # (chars_seq_len, batch_size)
            # word_pad_mask = torch.zeros_like(dec_in_char_seq, dtype=torch.bool, device=self.device).transpose_(0,1)
            word_pad_mask = None
            curve_pad_mask = None

            
            next_tokens_logits = self.model.decode(
                dec_in_char_seq, encoded, curve_pad_mask, word_pad_mask).transpose_(0, 1)[0, -1]
            if self.logit_processor:
                next_tokens_logits = self.logit_processor.process(
                    next_tokens_logits, cur_partial_hypothesis)
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1)
            topk_continuations = next_tokens_logproba.topk(self.beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                # Convert tesors to loat and int to avoid memory leakage.
                token_score = float(token_score)
                token_idx = int(token_idx)

                # Skipping tokens with prob = 0 (log_prob = -inf).
                # Theese tokens apper because even if there's less 
                # then `beamsize` tokens with non-zero probs
                # topk()  will still return exactly `beamsize` tokens. 
                # There are two sourses of zero prob: 
                # 1. Model is extremely confident (maybe overconfident) 
                #    that a certain token is impossible with a given prefix.
                # 2. Masking out unallowed tokens makes their prob = 0.
                if token_score == float('-inf'):
                    continue

                # score - нормализованная разность log_softmax всех токенов.
                # Разность, а не сумма, потому что heapq - мин-куча. 
                old_denorm_score = cur_partial_score * len(cur_partial_hypothesis)**self.normalization_factor
                new_score = (old_denorm_score - token_score) / (len(cur_partial_hypothesis) + 1)**self.normalization_factor

                new_hypothesis = cur_partial_hypothesis + [token_idx]
                new_item = (new_score, new_hypothesis)

                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= self.max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > self.beamsize:
                partial_hypotheses = heapq.nsmallest(self.beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = [self.tokenizer.decode(final_token_list[1:-1]) for final_token_list in final_token_lists]
        result = list(zip(final_scores, final_texts))
        result.sort()

        return result if self.return_hypotheses_n is None else result[:self.return_hypotheses_n]




# ! Note ! 
# GreedyGeneratorBatched Pros:
# * produces exactly the same results as GreedyGenerator
# * is more efficient (it's batched :) )
# GreedyGeneratorBatched Cons:
# * Doesn't support vocab masking yet
# * Has a different interface
class GreedyGeneratorBatched(WordGenerator):
    def __init__(
        self,
        model: EncoderDecoderTransformerLike,
        tokenizer: CharLevelTokenizerv2,
        device,
        logit_processor: Optional[LogitProcessor] = None,
        max_steps_n: int = 35,
    ):
        super().__init__(model, tokenizer, device, logit_processor)
        self.max_steps_n = max_steps_n

    @torch.inference_mode()
    def _generate(self, encoder_in: Tensor, encoder_in_pad_mask: Tensor) -> List[Tuple[float, str]]:
        # We suppose that BATCH_FIRST is False. Note that setting `BATCH_FIRST`
        # to True won't be the only step to make the code 
        # work with a model that expects batch_first data.
        BATCH_FIRST = False 
        # batch_size_dim = 0 if BATCH_FIRST else 1
        char_seq_len_dim = 1 if BATCH_FIRST else 0
        batch_size = encoder_in_pad_mask.size(0)  # mask is always batch_first
        dec_in_token_ids = torch.full((1, batch_size,), self.tokenizer.char_to_idx['<sos>'],
                                         device=self.device, dtype=torch.int32)
        sequence_finish_statuses = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_probs = torch.zeros(batch_size, device=self.device)
        eos_token_id = self.tokenizer.char_to_idx['<eos>']
        pad_token_id = self.tokenizer.char_to_idx['<pad>']
      
        encoder_in = move_encoder_in_to_device(encoder_in, self.device)
        encoder_in_pad_mask = encoder_in_pad_mask.to(self.device)
        encoded = self.model.encode(encoder_in, encoder_in_pad_mask)

        for _ in range(self.max_steps_n):
            # decoder_output.shape = char_seq_len x batch_size x n_tokens

            tgt_pad_mask = (dec_in_token_ids == pad_token_id).T

            next_tokens_logits = self.model.decode(
                dec_in_token_ids, encoded, encoder_in_pad_mask, tgt_pad_mask)[-1]  # shape = batch_size x n_tokens 
            
            # next_tokens_logits = self._mask_out_unallowed_ids(
            #     dec_in_char_seq.squeeze(BATCH_SIZE_DIM).tolist(),
            #     next_tokens_logits)

            # ! Note !  We don't really need to perform softmax since argmax would be the same.
            # It's only needed to return proper log probabilities (that can be used to output probabilities).
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1) # shape = batch_size x n_tokens
            next_tokens = next_tokens_logproba.argmax(dim=-1)  # shape = batch_size
            
            # Finished sequences should only be extended by pad tokens.
            next_tokens = next_tokens * ~sequence_finish_statuses + pad_token_id * sequence_finish_statuses

            # Our model never predict <pad> and actually, pad_oken_id is out of range of model's output.
            # Thus we need to add log_probs only for unfinishedsequences
            unfinished_indices = torch.nonzero(~sequence_finish_statuses, as_tuple=True)[0]
            log_probs[unfinished_indices] += next_tokens_logproba[unfinished_indices, next_tokens[unfinished_indices]]

            dec_in_token_ids = torch.cat([dec_in_token_ids, next_tokens.unsqueeze(char_seq_len_dim)], dim=char_seq_len_dim)
          
            sequence_finish_statuses |= next_tokens == eos_token_id
            if sequence_finish_statuses.all():
                break

        return dec_in_token_ids, log_probs

    def __call__(self, encoder_in, encoder_in_pad_mask) -> List[Tuple[float, str]]:
        return self._generate(encoder_in, encoder_in_pad_mask)


def generator_factory(generator_name: str,
                      model: EncoderDecoderTransformerLike,
                      subword_tokenizer: CharLevelTokenizerv2,
                      device: str,
                      logit_processor: Optional[LogitProcessor] = None,
                      generator_params: Optional[dict] = None
                      ) -> WordGenerator:
    if generator_params is None:
        generator_params = {}
    if generator_name == 'greedy':
        return GreedyGenerator(model, subword_tokenizer, device, logit_processor, **generator_params)
    elif generator_name == 'beam':
        return BeamGenerator(model, subword_tokenizer, device, logit_processor, **generator_params)
    else:
        raise ValueError(f"Unknown generator: {generator_name}")

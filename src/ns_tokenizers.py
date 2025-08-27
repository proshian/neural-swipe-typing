from typing import List, Set
import json


ALL_CYRILLIC_LETTERS_ALPHABET_ORD = [
    'а', 'б', 'в', 'г', 'д', 'е', 'ë', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'
]


class CharLevelTokenizerv2:
    """
    Tokenizes a word into a list of integers.

    Guarantees that <sos> and <pad> are tokens with `vocab_len - 1` and
    `vocab_len - 2` indices respectively. The model never needs to 
    predict <sos> and <pad> tokens. Since theese tokens have the biggest ids
    neuron_index is equal to token_id. Otherwise we would need a mapping
    from neuron_index to token_id
    """
    def __init__(self, vocab_path):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.max_word_len = None  # is set in _build_vocab
        # ! I don't think we will need <unk>, but it
        # doesn't lead to any problems. Decided to keep it.
        self.special_tokens = ["<eos>", "<unk>", "<pad>", "<sos>"]
        self._build_vocab(vocab_path)

    def _build_vocab(self, vocab_path):
        self.max_word_len = 0
        unique_chars = set()

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = f.read().split("\n")
            for word in vocab:
                self.max_word_len = max(self.max_word_len, len(word) + 2)  # + <sos> and <eos>
                for char in word:
                    unique_chars.add(char)
                    
        unique_chars_list = sorted(list(unique_chars)) + self.special_tokens
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars_list)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars_list)}

    def encode(self, word: str) -> List[int]:
        """
        Tokenizes a word into a list of integers.
        The <sos> and <eos> tokens are added.
        """
        tokenized_word = []
        tokenized_word.append(self.char_to_idx["<sos>"])
        for char in word:
            default: int = self.char_to_idx['<unk>']
            tokenized_word.append(self.char_to_idx.get(char, default))
        tokenized_word.append(self.char_to_idx["<eos>"])
        return tokenized_word
    
    def decode(self, token_seq):
        """
        Decodes a tokenized word into a string.
        """
        return "".join([self.idx_to_char[int(idx)] for idx in token_seq])


class KeyboardTokenizer:
    def __init__(self, json_path: str):
        with open(json_path, encoding='utf-8') as f:
            json_obj = json.load(f)
        self.label_to_idx = {ch: idx for idx, ch in enumerate(json_obj['labels'])}
        self.idx_to_label = {idx: ch for ch, idx in self.label_to_idx.items()}
        self.all_special_tokens = json_obj['special_tokens']

    def get_token(self, char):
        return self.label_to_idx.get(char, self.label_to_idx['<unk>'])

    def get_all_non_special_tokens(self) -> Set[str]:
        return set(self.label_to_idx.keys()) - set(self.all_special_tokens)

    def get_all_non_special_token_ids(self) -> Set[int]:
        return set(self.get_token(lbl) for lbl in self.get_all_non_special_tokens())
    
    def __len__(self):
        return len(self.label_to_idx)

from typing import List, Tuple, Dict
import json
import copy


def get_label_to_key_map(kb_keys: dict,
                         substitutions: dict = None,
                         copy_keys: bool = True) -> Dict[str, dict]:
    """
    Given a list of keys in kb_keys returns a dict where keys are labels
    and values are keys. If a label is absent in kb_keys, the function
    uses a label from substitutions dict instead. If there is no such
    label in substitutions, the function prints a warning.

    Arguments:
    ----------
    substitutions: dict
        Dict with keys being labels that are may be absent in grid
        and values being labels that should be used instead. 
        For example, {'ъ': 'ь', 'ё': 'е'}. If there is no 'ё' in grid,
        users swipes over 'е' instead.
    copy_keys: bool
        If True, the values in the returned dict are deepcopies of the
        keys in kb_keys. If False, the values are the keys themselves.
    """
    if substitutions is None:
        substitutions = {'ъ': 'ь', 'ё': 'е'}

    label2key = {}
    for key in kb_keys:
        if 'label' not in key:
            continue
        label2key[key['label']] = key if not copy_keys else copy.deepcopy(key)
    for potentially_missing_label in substitutions.keys():
        if potentially_missing_label not in label2key:
            if not potentially_missing_label in substitutions:
                print(f"Warning: Character '{potentially_missing_label}' not found in label2key")
                continue
            label2key[potentially_missing_label] = (
                label2key[substitutions[potentially_missing_label]])
    return label2key


def get_key_centers(target_word: str,
                    label_to_key: Dict[str, dict],
                    absent_chars_on_keyboard: Tuple[str] = ()) -> List[Tuple[int, int]]:
    """
    Arguments:
    ----------
    target_word: str
        The word being typed.
    label_to_key: Dict[str, dict]
        Mapping from character labels to keyboard key metadata.
    absent_chars_on_keyboard: Tuple[str]
        Characters that are not expected to be found on the keyboard.
    """
    key_centers = []
    for char in target_word:
        if char not in label_to_key:
            if char not in absent_chars_on_keyboard:
                print(f"Warning: Character '{char}' not found in label2key")
            continue
        key_centers.append(get_kb_key_center(label_to_key[char]['hitbox']))
    return key_centers



def get_kb_key_center(hitbox: Dict[str, int]) -> Tuple[float, float]:
    x = hitbox['x'] + hitbox['w'] / 2
    y = hitbox['y'] + hitbox['h'] / 2
    return x, y

def distance(x1, y1, x2, y2) -> float:
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5




def get_wh(grid: dict) -> tuple:
    return grid['width'], grid['height']

def get_gname_to_wh(gname_to_grid: Dict[str, dict]):
    return {gname: get_wh(grid)
            for gname, grid in gname_to_grid.items()}

def get_kb_label(key: dict) -> str:
    if 'label' in key:
        return key['label']
    if 'action' in key:
        return key['action']
    raise ValueError("Key has no label or action property")

def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]
    
def get_grid_name_to_grid(grid_name_to_grid__path: str, 
                          allowed_gnames = ("default", "extra")) -> dict:
    # In case there will be more grids in "grid_name_to_grid.json"
    grid_name_to_grid = {
        grid_name: get_grid(grid_name, grid_name_to_grid__path)
        for grid_name in allowed_gnames
    }
    return grid_name_to_grid


def get_avg_half_key_diag(grid: dict, 
                          allowed_keys: List[str]) -> float:
    hkd_list = []
    for key in grid['keys']:
        label = get_kb_label(key)
        if label not in allowed_keys:
            continue
        hitbox = key['hitbox']
        kw, kh = hitbox['w'], hitbox['h']
        half_key_diag = (kw**2 + kh**2)**0.5 / 2
        hkd_list.append(half_key_diag)
    return sum(hkd_list) / len(hkd_list)

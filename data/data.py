import os
import hashlib
import json
from pathlib import Path
from typing import Tuple, Union, Dict, Any
from core.interfaces import ITokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
from torch import Tensor
from tqdm import tqdm


CACHE_VERSION = 1


def compute_tokenizer_fingerprint(tokenizer: ITokenizer) -> str:
    """Compute a deterministic fingerprint of the tokenizer vocabulary and config."""
    vocab_items = sorted(tokenizer._token_to_id.items())
    special = {
        'pad': tokenizer.special_tokens.pad_id,
        'sos': tokenizer.special_tokens.sos_id,
        'eos': tokenizer.special_tokens.eos_id,
    }
    if tokenizer.special_tokens.blank_id is not None:
        special['blank'] = tokenizer.special_tokens.blank_id
    
    data = {
        'vocab': vocab_items,
        'special_tokens': special,
        'vocab_size': tokenizer.vocab_size,
    }
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def compute_dataset_fingerprint(data_path: Union[str, Path], clean_key: str, dist_key: str) -> str:
    """Compute a fingerprint of the dataset (row count + column names + sample hash)."""
    df = pd.read_csv(data_path, nrows=1000)  # Sample first 1000 rows for speed
    sample_data = df[clean_key].tolist() + df[dist_key].tolist()
    sample_str = '|'.join(sample_data)
    
    data = {
        'rows': len(df),
        'columns': list(df.columns),
        'clean_key': clean_key,
        'dist_key': dist_key,
        'sample_hash': hashlib.md5(sample_str.encode('utf-8')).hexdigest()[:16],
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def compute_config_fingerprint(max_len: int, ratio: float, dist_key: str, clean_key: str) -> str:
    """Compute fingerprint of preprocessing configuration."""
    data = {
        'max_len': max_len,
        'ratio': ratio,
        'dist_key': dist_key,
        'clean_key': clean_key,
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def get_cache_fingerprint(
        tokenizer: ITokenizer,
        data_path: Union[str, Path],
        max_len: int,
        ratio: float,
        dist_key: str,
        clean_key: str
        ) -> Dict[str, str]:
    """Compute combined fingerprint for cache validation."""
    return {
        'version': CACHE_VERSION,
        'tokenizer': compute_tokenizer_fingerprint(tokenizer),
        'dataset': compute_dataset_fingerprint(data_path, clean_key, dist_key),
        'config': compute_config_fingerprint(max_len, ratio, dist_key, clean_key),
    }


def validate_cache(cache_path: Union[str, Path], expected_fingerprint: Dict[str, str]) -> bool:
    """Validate that cache matches expected fingerprint."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    
    try:
        cached = torch.load(cache_path, weights_only=False)
        cached_fp = cached.get('fingerprint')
        if cached_fp is None:
            print(f"Cache {cache_path} missing fingerprint - invalidating")
            return False
        
        for key, expected_value in expected_fingerprint.items():
            if cached_fp.get(key) != expected_value:
                print(f"Cache fingerprint mismatch on '{key}': expected {expected_value}, got {cached_fp.get(key)}")
                return False
        
        print(f"Cache fingerprint validated: {cache_path}")
        return True
    except Exception as e:
        print(f"Cache validation error: {e} - invalidating")
        return False


class ArabicData(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            tokenizer: ITokenizer,
            pad_idx: int,
            max_len: int,
            ratio: float,
            dist_key: str,
            clean_key: str
            ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len + 2
        self.dist_key = dist_key
        self.clean_key = clean_key
        self.pad_idx = pad_idx
        self.max_dist_len = int(ratio * max_len) + max_len
        self.df = pd.read_csv(data_path)
        missing = [
            key for key in [self.clean_key, self.dist_key]
            if key not in self.df.columns
        ]
        if missing:
            raise KeyError(
                f'Missing required CSV columns {missing}. Available columns: '
                f'{list(self.df.columns)}'
            )

        cache_path = Path(data_path).with_suffix('.tokenized.pt')
        fingerprint = get_cache_fingerprint(tokenizer, data_path, max_len, ratio, dist_key, clean_key)
        
        if cache_path.exists() and validate_cache(cache_path, fingerprint):
            print(f"Loading pre-tokenized data from {cache_path}...")
            cached = torch.load(cache_path, weights_only=False)
            self.clean_tokenized = cached['clean']
            self.distorted_tokenized = cached['distorted']
            print(f"Loaded {len(self.clean_tokenized)} samples.")
        else:
            if cache_path.exists():
                print(f"Cache invalid or missing fingerprint - rebuilding...")
            else:
                print(f"Pre-tokenizing {len(self.df)} samples (first run only)...")
            clean_texts = self.df[self.clean_key].tolist()
            distorted_texts = self.df[self.dist_key].tolist()
            
            # Tokenize with progress bar
            clean_tokenized = []
            distorted_tokenized = []
            for i in tqdm(range(len(clean_texts)), desc="Tokenizing clean"):
                clean_tokenized.append(self.tokenizer.tokenize(
                    clean_texts[i], add_sos=True, add_eos=True
                ))
            for i in tqdm(range(len(distorted_texts)), desc="Tokenizing distorted"):
                distorted_tokenized.append(self.tokenizer.tokenize(
                    distorted_texts[i], add_sos=True, add_eos=True
                ))
            
            self.clean_tokenized = clean_tokenized
            self.distorted_tokenized = distorted_tokenized
            
            print(f"Saving pre-tokenized data to {cache_path}...")
            torch.save({
                'clean': self.clean_tokenized,
                'distorted': self.distorted_tokenized,
                'fingerprint': fingerprint,
            }, cache_path)
            print("Pre-tokenization complete and cached.")

    def pad(self, line: list, max_len: int) -> Tuple[list, int]:
        length = len(line)
        diff = max_len - length
        assert diff >= 0
        return line + [self.pad_idx] * diff, diff

    def _get_clean(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.clean_tokenized[idx]
        mask = [False] * len(item)
        item, diff = self.pad(item, self.max_len)
        mask += [True] * diff
        item = torch.LongTensor(item)
        mask = torch.BoolTensor(mask)
        return item, mask

    def _get_distorted(self, idx: int) -> Tuple[Tensor, Tensor]:
        item = self.distorted_tokenized[idx]
        mask = [False] * len(item)
        item, diff = self.pad(item, self.max_dist_len + 1)
        mask += [True] * diff
        item = torch.LongTensor(item)
        mask = torch.BoolTensor(mask)
        return item, mask

    def __getitem__(self, idx: int):
        clean, clean_mask = self._get_clean(idx)
        distorted, distorted_mask = self._get_distorted(idx)
        return distorted, clean, distorted_mask, clean_mask

    def __len__(self):
        return self.df.shape[0]


def get_train_loader(
        data_path,
        tokenizer,
        max_len,
        ratio,
        batch_size,
        rank,
        world_size,
        dist_key,
        clean_key,
        num_workers=0,
        pin_memory=False
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio,
        dist_key=dist_key,
        clean_key=clean_key
    )
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=world_size,
            drop_last=True,
            shuffle=True
        )
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True
    )


def get_test_loader(
        data_path,
        tokenizer,
        batch_size,
        max_len,
        ratio,
        dist_key,
        clean_key,
        num_workers=0,
        pin_memory=False
        ):
    dataset = ArabicData(
        data_path=data_path,
        tokenizer=tokenizer,
        pad_idx=tokenizer.special_tokens.pad_id,
        max_len=max_len,
        ratio=ratio,
        dist_key=dist_key,
        clean_key=clean_key
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False
    )


def get_train_test_loaders(args, rank: int, tokenizer: ITokenizer) -> tuple:
    assert os.path.exists(args.train_path), \
        f'{args.train_path} does not exist!'
    assert os.path.exists(args.test_path), \
        f'{args.test_path} does not exist!'
    
    num_workers = getattr(args, 'num_workers', 0)
    pin_memory = getattr(args, 'pin_memory', False)
    
    train_loader = get_train_loader(
        data_path=args.train_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        ratio=args.distortion_ratio,
        batch_size=args.batch_size,
        rank=rank,
        world_size=args.n_gpus,
        dist_key=args.dist_key,
        clean_key=args.clean_key,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = get_test_loader(
        data_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        ratio=args.distortion_ratio,
        dist_key=args.dist_key,
        clean_key=args.clean_key,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, test_loader
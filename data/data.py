import os
from pathlib import Path
from typing import Tuple, Union
from core.interfaces import ITokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
from torch import Tensor
from tqdm import tqdm


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
        if cache_path.exists():
            print(f"Loading pre-tokenized data from {cache_path}...")
            cached = torch.load(cache_path, weights_only=False)
            self.clean_tokenized = cached['clean']
            self.distorted_tokenized = cached['distorted']
            print(f"Loaded {len(self.clean_tokenized)} samples.")
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
                'distorted': self.distorted_tokenized
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
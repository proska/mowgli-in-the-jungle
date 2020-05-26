import os
import pickle
import pathlib
from logging import Logger
from typing import Dict, Tuple, Type, List, Optional

import torch
import tqdm
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTConfig, OpenAIGPTLMHeadModel, \
    OpenAIGPTTokenizer, BertConfig, BertForMaskedLM, BertTokenizer, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, \
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig

import logging


logger: Logger = logging.getLogger(__name__)


MODEL_CLASSES: Dict[str, Tuple[Type[PretrainedConfig], Type[PreTrainedModel], Type[PreTrainedTokenizer]]] = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class BlockTextDataset(Dataset):
    def __init__(self, tokenizer: Type[PreTrainedTokenizer],
                 file_path: str, block_size=512):

        file_path: pathlib.Path = pathlib.Path(file_path)
        assert file_path.is_file(), f'{file_path} is not file'

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        logger.info("Creating tokens from dataset file at %s", file_path)

        self.examples = []
        with open(str(file_path), encoding="utf-8") as f:
            text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            # Truncate in block of block_size
            for i in tqdm.tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        file_path: pathlib.Path = pathlib.Path(file_path)

        assert file_path.is_file(), f'{file_path} is not file'
        logger.info("Creating features from dataset file at %s", file_path)

        with open(str(file_path), encoding="utf-8") as f:
            lines = [line for line in tqdm.tqdm(f.readlines()) if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines,
                                                    add_special_tokens=True,
                                                    max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.tensor(
            self.examples[i],
            dtype=torch.long
        )


class LineByLineTextOnlyDataset(Dataset):
    def __init__(self, file_path: str):
        file_path: pathlib.Path = pathlib.Path(file_path)

        assert file_path.is_file(), f'{file_path} is not file'
        logger.info("Creating features from dataset file at %s", file_path)

        with open(str(file_path), encoding="utf-8") as f:
            lines = [line for line in tqdm.tqdm(f.readlines()) if (len(line) > 0 and not line.isspace())]

        self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int) -> str:
        return self.examples[i]


class BlockTextOnlyDataset(Dataset):
    def __init__(self, tokenizer: Type[PreTrainedTokenizer],
                 file_path: str, block_size=512):

        file_path: pathlib.Path = pathlib.Path(file_path)
        assert file_path.is_file(), f'{file_path} is not file'

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        logger.info("Creating tokens from dataset file at %s", file_path)

        self.examples = []
        with open(str(file_path), encoding="utf-8") as f:
            text: str = f.read()

        tokenized_text: List[str] = tokenizer.tokenize(text)

        # Truncate in block of block_size
        for i in tqdm.tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):
            self.examples.append(
                " ".join(tokenized_text[i: i + block_size])
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

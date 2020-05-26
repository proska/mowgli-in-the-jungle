import pathlib
from typing import List, Optional, Union, Dict, Any, Type

import IPython
import lazy_import
from tqdm import tqdm

transformers = lazy_import.lazy_module('transformers')
pl = lazy_import.lazy_module('pytorch_lightning')
np = lazy_import.lazy_module('numpy')
torch = lazy_import.lazy_module('torch')
nn = lazy_import.lazy_module('torch.nn')
torch_data = lazy_import.lazy_module('torch.utils.data')
AdamW = lazy_import.lazy_module('torch.optim.AdamW')

import mowgli.classes
from mowgli.predictor.PytorchLigthningPredictor import PytorchLightningPredictor
from mowgli.predictor.utils import ClassificationDataset
from mowgli.utils.LM.SentenceEmbedder import SentEmbedder
from mowgli.utils.graphs.KGNetworkx import NxKG

import logging

logger = logging.getLogger(__name__)


class BlockTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Type[transformers.PreTrainedTokenizer],
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



class ModelTrainer(pl.LightningModule):
    def __init__(self, max_length: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_sent_len = max_length
        self.LM = SentEmbedder(name='roberta-base', token_pooling='mean')
        self.kg = NxKG.load(path='')


    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.LM(**tokens)

    def collate(self, sents, do_mlm: bool = False):
        tokens = self.LM.tokenize(sents, max_length=self.max_sent_len)
        # if self.tokenizer._pad_token is None:
        #     padded = pad_sequence(token_ids, batch_first=True)
        # else:
        #     padded = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # if do_mlm:
        #     inputs, labels = self._mask_tokens(padded)
        return tokens

    def _mask_tokens(self, tokens):
        pass
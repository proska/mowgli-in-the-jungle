import os
import pdb
from typing import Dict, Union, Optional, NoReturn, List, Tuple

import IPython
import omegaconf
import logging
import pytorch_lightning as pl
import torch
import torch.optim
import torch.utils
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel, AdamW, get_linear_schedule_with_warmup
from .HuggingFaceUtils import MODEL_CLASSES, LineByLineTextDataset, BlockTextDataset, LineByLineTextOnlyDataset, \
    BlockTextOnlyDataset

logger = logging.getLogger(__name__)


class HFLanguageModel(pl.LightningModule):

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.hparams: Dict[str, Union[int, float, str, bool, torch.Tensor]] = {}
        self._setup_hparams(config)
        self._check_hparams()

        self.lm_config: Optional[PretrainedConfig] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.lang_model: Optional[PreTrainedModel] = None
        self._setup_hf_language_model()

        self.hparams['train_batch_size'] = self.hparams['per_gpu_train_batch_size'] * max(1, self.hparams['n_gpu'])
        self.hparams['eval_batch_size'] = self.hparams['per_gpu_eval_batch_size'] * max(1, self.hparams['n_gpu'])

    def _setup_hf_language_model(self) -> NoReturn:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.hparams['model_type']]

        # setup HF config params
        if self.hparams['config_name']:
            self.lm_config = config_class.from_pretrained(self.hparams['config_name'],
                                                          cache_dir=self.hparams['cache_dir'])
        elif self.hparams['model_name_or_path']:
            self.lm_config = config_class.from_pretrained(self.hparams['model_name_or_path'],
                                                          cache_dir=self.hparams['cache_dir'])
        else:
            self.lm_config = config_class()

        # setup tokenizer
        if self.hparams['tokenizer_name']:
            self.tokenizer = tokenizer_class.from_pretrained(self.hparams['tokenizer_name'],
                                                             cache_dir=self.hparams['cache_dir'])
        elif self.hparams['model_name_or_path']:
            self.tokenizer = tokenizer_class.from_pretrained(self.hparams['model_name_or_path'],
                                                             cache_dir=self.hparams['cache_dir'])
        else:
            raise ValueError(
                f"You are instantiating a new {tokenizer_class.__name__} tokenizer. "
                "This is not supported, but you can do it from another script, save it,"
                "and load it from here, using tokenizer_name=<name>"
            )

        # setup max len for the sequence
        if 'max_sent_len' not in self.hparams or self.hparams['max_sent_len'] <= 0:
            self.hparams['max_sent_len'] = self.tokenizer.max_len
            # Our input block size will be the max possible for the model
        else:
            self.hparams['max_sent_len'] = min(self.hparams['max_sent_len'], self.tokenizer.max_len)

        if self.hparams['model_name_or_path']:
            self.lang_model = model_class.from_pretrained(
                self.hparams['model_name_or_path'],
                from_tf=bool(".ckpt" in self.hparams['model_name_or_path']),
                config=self.lm_config,
                cache_dir=self.hparams['cache_dir'],
            )
        else:
            logger.info("Training new model from scratch")
            self.lang_model = model_class(config=self.lm_config)

    def _setup_hparams(self, config: omegaconf.dictconfig.DictConfig, prefix: str = '') -> NoReturn:
        key = lambda k: (f'{prefix}.' if prefix != '' else '') + f'{k}'
        for k, v in dict(config).items():
            if any([isinstance(v, t) for t in [int, float, str, bool, torch.Tensor]]):
                self.hparams[key(k)] = v
            elif v is None:
                self.hparams[key(k)] = ''
            elif isinstance(v, omegaconf.dictconfig.DictConfig):
                self._setup_hparams(v, prefix=k)
            else:
                raise ValueError(f'invalid config type: [{k}]={v} with type {type(v)}')

    def _check_hparams(self) -> NoReturn:
        if (self.hparams['model_type'] in ["bert", "roberta", "distilbert", "camembert"]) and \
                (not self.hparams['mlm']):
            raise ValueError(
                "BERT and RoBERTa-like models do not have LM heads but masked LM heads. "
                "They must be run using the --mlm flag (masked language modeling)."
            )
        if self.hparams['eval_data_file'] is None and self.hparams['do_eval']:
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                "or remove the --do_eval argument."
            )

    def _collate(self, examples: Tuple[List[str]]) -> Dict[str, torch.Tensor]:
        # assert len(examples) == 1, f'Check: ({len(examples)})'
        # assert isinstance(examples, tuple), f'Check: ({type(examples)})'
        # assert (len(examples[0]) == self.hparams['train_batch_size']) or \
        #        (len(examples[0]) == self.hparams['eval_batch_size']), f'Check: {len(examples[0])}'
        token_ids = self.tokenizer.batch_encode_plus(
            examples,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.hparams['max_sent_len']
        )["input_ids"]

        if self.tokenizer._pad_token is None:
            padded = pad_sequence(token_ids, batch_first=True)
        else:
            padded = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        inputs, labels = self._mask_tokens(padded) if self.hparams['mlm'] else (padded, padded)

        return {
            'inputs': inputs,
            'labels': labels,
        }

    ##################################################################

    # def on_save_checkpoint(self, checkpoint):
    #     # 99% of use cases you don't need to implement this method
    #     checkpoint['hparams'] = self.hparams
    #     checkpoint['lm_config'] = self.hparams
    #     checkpoint['tokenizer'] = self.hparams
    #     checkpoint['lang_model'] = self.hparams
    #     self.lm_config: Optional[PretrainedConfig] = None
    #     self.tokenizer: Optional[PreTrainedTokenizer] = None
    #     self.lang_model: Optional[PreTrainedModel] = None

    ##################################################################

    def _dataloader(self, file_path: str) -> torch.utils.data.Dataset:
        # file_path = self.hparams['eval_data_file'] if evaluate else self.hparams['train_data_file']
        if self.hparams['line_by_line']:
            return LineByLineTextOnlyDataset(file_path=file_path)
        else:
            assert self.tokenizer is not None
            return BlockTextOnlyDataset(self.tokenizer,
                                        file_path=file_path, block_size=self.hparams['max_sent_len'])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._dataloader(file_path=self.hparams['train_data_file']),
            batch_size=self.hparams['train_batch_size'],
            num_workers=self.hparams['cpu_limit'],
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._dataloader(file_path=self.hparams['eval_data_file']),
            batch_size=self.hparams['train_batch_size'],
            num_workers=self.hparams['cpu_limit'],
            collate_fn=self._collate,
        )

    ##################################################################

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.lang_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams['weight_decay'],
            },
            {"params": [p for n, p in self.lang_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))

        # t_total = len(self.train_dataloader()) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]
        if self.hparams['max_steps'] > 0:
            t_total = self.hparams['max_steps']
        else:
            t_total = (len(self.train_dataloader()) //
                       self.hparams['gradient_accumulation_steps'] *
                       self.hparams['num_train_epochs'])

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams['warmup_steps'], num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
                self.hparams['model_name_or_path']
                and os.path.isfile(os.path.join(self.hparams['model_name_or_path'], "optimizer.pt"))
                and os.path.isfile(os.path.join(self.hparams['model_name_or_path'], "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.hparams['model_name_or_path'], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.hparams['model_name_or_path'], "scheduler.pt")))

        return [optimizer], [scheduler]

    ##################################################################
    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        # pdb.set_trace()
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.hparams['mlm_probability'])
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = batch['inputs']
        labels = batch['labels']

        lm_out = (self.lang_model(inputs, masked_lm_labels=labels)
                  if self.hparams['mlm'] else
                  self.lang_model(inputs, labels=labels))
        loss = lm_out[0]  # model outputs are always tuple in transformers (see doc)

        tqdm_dict = {'mlm_train_loss': loss}
        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            # 'log': tqdm_dict
        }
        return output

    ##################################################################

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs = batch['inputs']
        labels = batch['labels']

        lm_out = (self.lang_model(inputs, masked_lm_labels=labels)
                  if self.hparams['mlm'] else
                  self.lang_model(inputs, labels=labels))
        loss = lm_out[0]  # model outputs are always tuple in transformers (see doc)

        tqdm_dict = {'mlm_val_loss': loss}
        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
        }
        return output

    ##################################################################

    def forward(self, *args, **kwargs):
        pass
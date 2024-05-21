import itertools
import json
import linecache
import math
import os
import pickle
import socket
import glob
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader

# from sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer, BertTokenizer, RobertaTokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right
from datasets import load_dataset

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


logger = getLogger(__name__)


def write_json(data,name):
    json_string=json.dumps(data,ensure_ascii=False)
    with open(name, "w",encoding = 'utf-8') as file:
        file.write(json_string)



def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
        logger.info("note: command line args may override some of these")
        model.config.update(pars)



ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5" or model_type == "mt5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )

class dpo_dataset(Dataset):
    def __init__(self,data_dir,split="train"):
        file=os.path.join(data_dir,f"data_{split}.json")
        self.data_list = load_dataset("json",data_files=file,split=split).shuffle()
    def __getitem__(self,index):
        # {"chosen": , "prompt": , "data_id": ,"rejected":}
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def map(self, func, **kwargs):
        return self


class RuCn_Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, padding=None, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang
        self.is_bert_based = self.tokenizer.cls_token is not None
        self.padding = padding if padding is not None else ("max_length" if self.tpu_num_cores is not None else "longest")
        ###
        if hasattr(self.data_args, "langid_map"):
            self.ru_mapped_data = self.data_args.langid_map.get('russian', None)#'▁<extra_id_63>'
            self.cn_mapped_data= self.data_args.langid_map.get('chinese_simplified', None)
        else:
            logger.error(f"Unknown langid: russian or chinese_simplified")


    def __call__(self, features) -> Dict[str, torch.Tensor]:
        init_token_id = self.pad_token_id
        batch = {}
        # get id from vocab
        init_token_ids=[]
        for f in features:
            if f['data_id']=='russian':
                lang_idx, mapped_token = self.ru_mapped_data
            elif f['data_id']=='chinese_simplified':
                lang_idx, mapped_token = self.cn_mapped_data
            else:
                logger.error(f"Unknown langid: {f['data_id']}")
            init_token_ids.append(self.tokenizer._convert_token_to_id(mapped_token))
        
        init_token_ids=torch.cat(init_token_ids,0)
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(features)
            input_ids, attention_mask, labels,rejected_input_ids,rejected_attention_mask = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
            )

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels, init_token_ids)
            rejected_decoder_input_ids= self._shift_right_t5(rejected_input_ids, init_token_ids)
            if self.data_args.use_langid_prefix:
                input_ids = self._shift_right_t5(input_ids, init_token_ids)

        elif self.is_bert_based:
            # bert based models will automatically add the [CLS] token
            pass
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)
                 
        batch = dict(
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            chosen_input_ids=decoder_input_ids,
            chosen_attention_mask=torch.ones_like(decoder_input_ids),
            chosen_labels=labels,
            rejected_input_ids=rejected_decoder_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            rejected_labels=rejected_input_ids,
            init_token_id=init_token_id
            )
         
        return batch

    def _shift_right_t5(self, input_ids, init_token_id):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = init_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]: 
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["prompt"] for x in batch],
            tgt_texts=[x["chosen"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        batch_encoding_data=batch_encoding.data
        if "rejected" in batch[0].keys():
            rejected_batch_encoding = self.tokenizer(
                [x['rejected'] for x in batch],
                max_length=self.data_args.max_target_length,
                padding=self.padding,  # TPU hack
                return_tensors="pt",
                **self.dataset_kwargs,)
            rejected_batch_encoding=rejected_batch_encoding.data
            rejected_batch_encoding["rejected_attention_mask"]=rejected_batch_encoding.pop("attention_mask")
            rejected_batch_encoding["rejected_input_ids"]=rejected_batch_encoding.pop("input_ids")
            batch_encoding_data.update(rejected_batch_encoding)

        return batch_encoding_data#(input_ids,attention_mask,label)


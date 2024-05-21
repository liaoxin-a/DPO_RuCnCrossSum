import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")
import logging
import os

import sys
import glob
import json
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import  is_main_process
from transformers.training_args import ParallelMode
from utils import (
    check_output_dir,
    save_json,
    use_task_specific_params,
    freeze_embeds,
    freeze_params,
    assert_all_frozen,
    write_json,
    dpo_dataset,
    RuCn_Seq2SeqDataCollator
)

from torch import nn
from tqdm import tqdm
from trl import DPOTrainer
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, List
logger = logging.getLogger(__name__)


def read_json(path):
    with open(path) as f:
        return json.load(f)



def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=84,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=84,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=84,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation / summarization."})
    length_penalty: Optional[float] = field(default=0.6, metadata={"help": "# length_penalty"})

    rouge_lang: Optional[str] = field(default=None, 
        metadata={"help": "# apply language specific tokenization and stemming (if available)"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    use_langid: bool = field(
        default=False,
        metadata={"help": "whether to use langid during training / inference. `tgt_lang` will be used for inference"}
    )
    langid_map_path: str = field(
        default=None,
        metadata={
            "help": "Path to the langid map (json file). "
            "This map will be used if there isnt any map present in model config."
        }
    )
    reinitialize_langid_embeddings: Optional[str] = field(
        default=None,
        metadata={
            "help": "Embeddings to use for reinitializing langid embeddings",
            "choices": ["random", "bos"]
        }    
    )
    add_langid_embeddings: Optional[str] = field(
        default=None,
        metadata={
            "help": "Add langid embeddings during each time step. (only decoder supported rn)",
            "choices": ["encoder", "decoder", "both"]
        }    
    )

    tie_word_embeddings: bool = field(
        default=False,
        metadata={
            "help": "tie input and output word embeddings"
        }    
    )
    use_langid_prefix: bool = field(
        default=False,
        metadata={
            "help": "Use langid as prefix for both input and output"
        }    
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Setup logging
    # training_args.device=device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "tie_word_embeddings": data_args.tie_word_embeddings
    }
    if data_args.max_target_length:
        config_kwargs.update({'max_length': data_args.max_target_length})
    if data_args.length_penalty: 
        config_kwargs.update({'length_penalty': data_args.length_penalty})

    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, **config_kwargs,
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False, cache_dir=model_args.cache_dir,local_files_only=True
    )


    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )


    if data_args.use_langid:
        langid_map = None
        if config.task_specific_params is not None:
            logger.info(f"Loading langid map from model config")
            langid_map = config.task_specific_params.get("langid_map", None)
                 
        if data_args.add_langid_embeddings is not None:
            config.task_specific_params = config.task_specific_params or {}
            config.task_specific_params.update(
                {
                    "add_langid_embeddings": data_args.add_langid_embeddings
                }
            )

        if langid_map is None:
            assert os.path.isfile(data_args.langid_map_path), "Invalid langid_map provided."
            logger.info(f"Loading langid map from {data_args.langid_map_path}")
            
            with open(data_args.langid_map_path) as f:
                langid_map = json.load(f)
                langid_map = {k: [i, v] for i, (k, v) in enumerate(langid_map.items())}
                config.task_specific_params = config.task_specific_params or {}
                config.task_specific_params.update(
                    {
                        "langid_map": langid_map,
                        # "lang_head_weight": data_args.lang_head_weight
                        
                    }
                )

                if isinstance(model, T5ForConditionalGeneration) and data_args.reinitialize_langid_embeddings is not None:
                    embedding_weights = model.shared.weight.clone().detach()
                    new_langid_embedding = nn.Embedding(len(langid_map), config.d_model)
                    new_langid_embedding.weight.data.normal_(mean=0.0, std=1.0)
                    langid_embedding_weights = new_langid_embedding.weight.clone().detach()

                    for i, v in langid_map.values():
                        id = tokenizer._convert_token_to_id(v)
                        embedding_weights[id] = (
                            embedding_weights[tokenizer.pad_token_id] if data_args.reinitialize_langid_embeddings == "bos" 
                            else langid_embedding_weights[i]
                        )
                        
                    new_embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
                    model.set_input_embeddings(new_embedding)

        setattr(data_args, "langid_map", langid_map)

    # use task specific params
    use_task_specific_params(model, data_args.task)


    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())
    

    # Get datasets !!
    # dataset_collator = Seq2SeqDataCollator(tokenizer, data_args, "max_length", training_args.tpu_num_cores)
    # dpo_train_dataset = load_dataset("json", data_files="dpo_data/data_train.json",split="train")
    # dpo_train_dataset = dpo_train_dataset.shuffle().select(range(51))
    
    # dpo_train_dataset=dpo_dataset("dataset/data_test.json",tokenizer, data_args,"train", "max_length", training_args.tpu_num_cores)
    # dataset_collator = RuCn_Seq2SeqDataCollator(tokenizer, data_args, "max_length", training_args.tpu_num_cores)
    # dpo_train_dataset=dpo_dataset(data_args.data_dir,"train")
    # print(dpo_train_dataset[:2] )



    def add_init_token(data):
        langid_map={
            "chinese_simplified": [
            40,
            "\u2581<extra_id_59>"
            ],
            "russian": [
            36,
            "\u2581<extra_id_63>"
            ],
            }
        target_lang=data['data_id']
        mapped_token= langid_map[target_lang][1]
        return {
            'prompt': data['prompt'],
            'chosen': mapped_token+data['chosen'],
            'rejected':  mapped_token+data['rejected']
        }

    
    search_str = os.path.join(data_args.data_dir, f'*train.json')
    src_files=glob.glob(search_str)
    train_file_path=os.path.join(data_args.data_dir, 'data_train.json')
    use_preprocessed_data=False
    for src in src_files:
        if 'preprocessed_' in src:
            train_file_path=src#use_preprocessed_data
            use_preprocessed_data=True
    
    logger.info("PreInit data")
    dpo_train_dataset = load_dataset("json", data_files=train_file_path,split="train")
    if not use_preprocessed_data:
        dpo_train_dataset=dpo_train_dataset.map(add_init_token,remove_columns='data_id')
        dataset_dic=dpo_train_dataset.to_dict()
        write_json(dataset_dic,"dataset/preprocessed_data_train.json")
    ####


    all_metrics = {}
    # dpo Training



    # training_args.max_length=data_args.max_source_length
    # training_args.max_prompt_length=data_args.max_source_length
    # training_args.max_target_length=data_args.max_target_length

    logger.info("*** Trainner ***")
    dpo_trainer = DPOTrainer(
        model, 
        ref_model=None, 
        beta=0.1, 
        train_dataset=dpo_train_dataset, 
        # data_collator=dataset_collator,
        max_length=data_args.max_source_length,
        max_prompt_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        tokenizer=tokenizer, 
        args=training_args)
    

    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    metrics["train_n_objs"] = data_args.n_train

    dpo_trainer.save_model()  # this also saves the tokenizer

    if dpo_trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        dpo_trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        tokenizer.save_pretrained(training_args.output_dir)
    if dpo_trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))
    
    return all_metrics


if __name__ == "__main__":
    main()

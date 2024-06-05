
import torch
import logging
import os
import sys
import glob
import json
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils import data
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    build_compute_metrics_fn,
    check_output_dir,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler


logger = logging.getLogger(__name__)

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
    is_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "is encoder decoder"},
    )
    tie_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "tie encoder decoder"},
    )



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
    eval_beams: Optional[int] = field(default=4, metadata={"help": "# num_beams to use for evaluation."})
    length_penalty: Optional[float] = field(default=0.6, metadata={"help": "# length_penalty"})
    no_repeat_ngram_size: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    upsampling_factor: Optional[float] = field(default=None, 
        metadata={"help": "# use data upsampling factor only when using multiple data files."}
    )
    multistage_upsampling_factors: Optional[List[float]] = field(default=None, 
        metadata={"help": "# two stage upsampling factors when using xlingual summarization"}
    )
    per_lang_batch_size: Optional[int] = field(default=None, 
        metadata={"help": "# effective batch size per language"}
    )
    rouge_lang: Optional[str] = field(default=None, 
        metadata={"help": "# apply language specific tokenization and stemming (if available)"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    do_preprocess: bool = field(
        default=False,
        metadata={"help": "whether to tokenize the dataset first"}
    )
    use_preprocessed_data: bool = field(
        default=False,
        metadata={"help": "whether to use preprocessed data from disk or tokenize dynamically"}
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
    minibatching: Optional[str] = field(
        default=None,
        metadata={
            "help": "Minibatching options",
            "choices": ["ignored", "fixed_src", "fixed_tgt"]
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

    dpo_data: bool = field(
        default=False,
        metadata={
            "help": "Use dpo dataset"
        }    
    )
    lang_head_weight: float = field(
        default=0.3,
        metadata={
            "help": "langid head weight when using langids"
        }
    )
    
    

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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
     
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args,training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args,training_args = parser.parse_args_into_dataclasses()


    check_output_dir(training_args)

    # Setup logging
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



    config_kwargs = {
        "tie_word_embeddings": data_args.tie_word_embeddings
    }
    if data_args.max_target_length:
        config_kwargs.update({'max_length': data_args.max_target_length})
    if data_args.eval_beams:
        config_kwargs.update({'num_beams': data_args.eval_beams})
    if data_args.length_penalty:
        config_kwargs.update({'length_penalty': data_args.length_penalty})
    if data_args.no_repeat_ngram_size:
        config_kwargs.update({'no_repeat_ngram_size': data_args.no_repeat_ngram_size})
    
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
        use_fast=False, cache_dir=model_args.cache_dir,
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


    model.config.decoder_start_token_id = tokenizer._convert_token_to_id(langid_map[data_args.tgt_lang][1])

    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams


    dataset_class = Seq2SeqDataset
    dataset_collator = Seq2SeqDataCollator(tokenizer, data_args, None, training_args.tpu_num_cores)

    # Get datasets
    eval_dataset=None
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            data_id=data_args.tgt_lang
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            data_id=data_args.tgt_lang
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer, data_args) if training_args.predict_with_generate else None
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        data_collator=dataset_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        
    )

    all_metrics = {}
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val", 
            max_length=data_args.val_max_target_length, 
            num_beams=data_args.eval_beams,
            length_penalty=data_args.length_penalty,
            no_repeat_ngram_size=data_args.no_repeat_ngram_size,
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix="test",
            max_length=data_args.test_max_target_length,
            num_beams=data_args.eval_beams,
            length_penalty=data_args.length_penalty,
            no_repeat_ngram_size=data_args.no_repeat_ngram_size,
        )
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                predictions = test_output.predictions
                predictions[predictions == -100] = tokenizer.pad_token_id
                test_preds = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics




if __name__ == "__main__":
    main()

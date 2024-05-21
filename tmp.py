from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    set_seed,
)
from datasets import load_dataset
import re
model_name_or_path="csebuetnlp/mT5_m2m_crossSum"
cache_dir="cache_dir"
tokenizer_name=None
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
# tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_name if tokenizer_name else model_name_or_path,
#         use_fast=False, cache_dir=cache_dir,
#     )
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
def test_init_token_id():
    cn_mapped_data= langid_map.get('chinese_simplified', None)
    lang_idx, mapped_token = cn_mapped_data
    str="你好吗？我爱你。"

    prompt_input_ids = tokenizer.encode(mapped_token+ str,add_special_tokens=False)
    chosen_input_ids = tokenizer.encode(str,add_special_tokens=False)
    init_token_id=tokenizer._convert_token_to_id(mapped_token)

    print("prompt_input_ids=",prompt_input_ids)
    print("chosen_input_ids=",chosen_input_ids)
    print("init_token_id=",init_token_id)

def test_model_inputids():
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""

    model_name = "csebuetnlp/mT5_m2m_crossSum"

    

    get_lang_id = lambda lang: tokenizer._convert_token_to_id(
    model.config.task_specific_params["langid_map"][lang][1]) 

    target_lang = "chinese_simplified" # for a list of available language names see below

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    


    output_ids = model.generate(
        input_ids=input_ids,
        decoder_start_token_id=get_lang_id(target_lang),
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4,
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(summary)


def test_dataset():
    dpo_train_dataset = load_dataset("json", data_files="dpo_data/data_train.json",split="train")
    # get_lang_id = lambda lang: tokenizer._convert_token_to_id(
    # model.config.task_specific_params["langid_map"][lang][1]) 
    def add_init_token(data):
        mapped_token= langid_map[data['data_id']][1]
        return {
            'prompt': data['prompt'],
            'chosen': mapped_token+data['chosen'],
            'rejected':  mapped_token+data['rejected']
        }
    dpo_train_dataset=dpo_train_dataset.map(add_init_token,remove_columns='data_id')
    print(dpo_train_dataset[:1])


if __name__=="__main__":
    test_dataset()




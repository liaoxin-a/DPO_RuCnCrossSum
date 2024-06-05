# DPO_RuCnCrossSum

## Setup
```
$ git clone https://github.com/liaoxin-a/DPO_RuCnCrossSum
$ python -m nltk.downloader punkt
$ pip install --upgrade -r requirements.txt
```

## Downloading data
The dataset files are organized in `.json` format . **Download the dataset from [here] https://drive.google.com/file/d/1tg4TcU3GQDZeDZ2uDHbVw3PM1HWmmkLx/view?usp=drive_link).**



## Optimization with DPO

To see the list of all available options related to optimization with DPO, do `python dpo_train.py -h`

## Evaluation

* See available evaluation options: `python evaluator.py -h`. 
 
For example, to compute `ROUGE` and `LaSE` scores on all language pairs of the CrossSum test set using a trained cross-lingual model, run the following:

```bash
python evaluator.py \
    --dataset_dir <path/to/dataset/directory> \
    --output_dir <path/to/output/directory> \
    --evaluation_type xlingual \
    --data_type test \
    --xlingual_summarization_model_name_or_path <path/to/model/directory>
```

More detailed examples can be found in [evaluate.sh](evaluate.sh)

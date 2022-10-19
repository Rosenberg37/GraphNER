# README

Code for "**Type-supervised sequence labeling based on the heterogeneous star graph for named entity recognition**".

![](./assets/framework.png)

## Setup

### Requirements

You can try to create environment as follows:

```bash
conda create --name GrapnNER python=3.9.13
conda activate GraphNER
pip install -r requirements.txt
```

or directly import conda environment on Windows as follows:

```bash
conda env create -f windows.yaml
```

or directly import conda environment on Linux as follows:

```bash
conda env create -f linux.yaml
```

### Datasets

Original source of datasets:

+ GENIA: http://www.geniaproject.org/genia-corpus
+ CoNLL03: https://data.deepai.org/conll2003.zip
+ WeiboNER: https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/Weibo

You can download our processed datasets from [here](https://drive.google.com/drive/folders/1yDHy91z_eFg7ElllntCWvae5BNX-LWzT?usp=sharing).

Data format:

```json
{
  "tokens": [
    "IL-2",
    "gene",
    "expression",
    "and",
    "NF-kappa",
    "B",
    "activation",
    "through",
    "CD28",
    "requires",
    "reactive",
    "oxygen",
    "production",
    "by",
    "5-lipoxygenase",
    "."
  ],
  "entities": [
    {
      "start": 14,
      "end": 15,
      "type": "protein"
    },
    {
      "start": 4,
      "end": 6,
      "type": "protein"
    },
    {
      "start": 0,
      "end": 2,
      "type": "DNA"
    },
    {
      "start": 8,
      "end": 9,
      "type": "protein"
    }
  ],
  "relations": {},
  "org_id": "ge/train/0001",
  "pos": [
    "PROPN",
    "NOUN",
    "NOUN",
    "CCONJ",
    "PROPN",
    "PROPN",
    "NOUN",
    "ADP",
    "PROPN",
    "VERB",
    "ADJ",
    "NOUN",
    "NOUN",
    "ADP",
    "NUM",
    "."
  ],
  "ltokens": [],
  "rtokens": []
}
```

The `ltokens` contains the tokens from the previous sentence. And The `rtokens` contains the tokens from the next sentence.

## Word vectors

For used word vectors including Chinese word2vec, Glove and Bio-word2vec, you can download from [here](https://drive.google.com/drive/folders/1KV0LBjkO7lwMgZ4SEeJ0O6oUUF6ImHuc?usp=sharing).

## Run

You can run the experiment on GENIA dataset as follows:

```bash
python main.py --dataset_name=genia --evaluate=test --concat --pretrain_select=dmis-lab/biobert-base-cased-v1.2 --word2vec_select=bio --batch_size=4 --epochs=5 --max_length=128 --pos_dim=50 --char_dim=50
```

You can run the experiment on weiboNER dataset as follows:

```bash
python main.py --dataset_name=weiboNER --evaluate=dev --evaluate=test --pretrain_select=bert-base-chinese --word2vec_select=chinese --batch_size=4 --epochs=5 --max_length=64
```

You can run the experiment on Conll2003 dataset as follows:

```bash
python main.py --dataset_name=conll2003 --evaluate=test --concat --pretrain_select=bert-base-cased --word2vec_select=glove --batch_size=4 --epochs=5 --max_length=128 --pos_dim=50 --char_dim=50
```

## Reference

If you have any questions related to the code or the paper or the copyright, please email `wenxr2119@mails.jlu.edu.cn`.


# COLLEGE
code & data for paper "Contrastive Language-Knowledge Graph Pre-training"
### Overview

## Download pretrained models
You can download pretrained COLLEGE models below.
| Model | Size | Pretraining Text | Pretraining Knowledge Graph | Download Link |
| ------------- | --------- | ---- | ---- | ---- |
| COLLEGE   | 360M parameters | Wikipedia | Wikidata | [model checkpoint](https://nlp.stanford.edu/projects/myasu/DRAGON/models/general_model.pt) |

## Re-train rhe COLLEGE
### 1. Download the data

Download the latest wiki dump (XML format):

```bash
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Download the knowledge graph (Wikidata5M):

```bash
wget -c https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1
tar -xzvf wikidata5m_transductive.tar.gz
```

Download the Wikidata5M entity & relation aliases:

```bash
wget -c https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1
tar -xzvf wikidata5m_alias.tar.gz
```

### 2. Preprocess the data

Preprocess wiki dump:

```bash
mkdir pretrain_data
# process xml-format wiki dump
python preprocess/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o pretrain_data/output -l --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 4
python preprocess/extract.py 4
# generate the (text, graph) pair data
python preprocess/data.py 32
```

### 3. Train COLLEGE


```bash
cd code/
python -m torch.distributed.launch --nproc_per_node n train_2hop.py —gpu_num=n # replace the $n$ as the gpu number
```
## Run the experiments

Download the datasets for the experiments in the paper: [Google Drive](https://drive.google.com/file/d/1UNXICdkB5JbRyS5WTq6QNX4ndpMlNob6/view?usp=sharing).

```bash
python download_gdrive.py 1UNXICdkB5JbRyS5WTq6QNX4ndpMlNob6 ./data.tar.gz
tar -xzvf data.tar.gz
cd finetune/
```

#### FewRel

```bash
python run_re.py --debug --gpu 0
```

#### Open Entity

```bash
python run_typing.py --debug --gpu 0
```

#### LAMA and LAMA-UHN

```bash
cd ../lama/
python eval_lama.py
```


## Citation
If you find our work helpful, please cite the following:

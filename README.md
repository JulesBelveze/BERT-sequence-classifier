# 🤗 BERT-Multi-Label-Classifier / Dockerized Inferer 🤗
Repository to fine-tune a BERT-base multi-label/multi-class classifier, based on _HuggingFace_ library. The repository includes a _Flask_ API wrapper for inference.

## Table of contents
* [Installation](#installation)
* [Organisation of files](#organisation-of-files)
* [Datasets](#datasets)
* [Models](#models)
  * [Multi-label-classifier](#multi-label-classifier)
  * [Multi-class-classifier](#multi-class-classifier)
* [Inference](#inference)
* [TODO](#todo)

## Installation
To install the repository please run the following command:
```
git clone https://github.com/JulesBelveze/BERT-multi-label-classifier.git
```
The repository uses _Poetry_ as a package manager (see full documentation [here](https://python-poetry.org/docs/#installation)). To install the required packages please run the following commands:
```
python3 -m venv .venv/bert-mlc
source .venv/bert-mlc/bin/activate
poetry install
```

## Organisation of files
* `models/`: folder containing custom models
* `utils/`: folder containing function utilities
* `main.py`: main file to run
* `train.py`: file containing the training procedure
* `eval.py`: file containing the evaluation procedure
* `app.py`: file containing the _Flask_ app
* `inferer.py`: file containing the model inferer
* `poetry.lock`: _Poetry_ file
* `pyproject.toml`: _Poetry_ file
* `requirements_inference.txt`: required packages for inference
* `Dockerfile`: file to run the API as a docker image

## Datasets
* **multi-class:** you can download it [here](https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv)
* **multi-label:** [Toxic Comment Classification Challenge | Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Models
We provide customisation of three different models: BERT, Roberta and Distilbert.
### 1. Multi-label-classifier
The model is an adaptation of  the `BertForSequenceClassification` model of [HuggingFace](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) to handle multi-label. The key modification here is the modification of loss function.
### 2. Multi-class-classifier
The model used is basically a MLP on top of a BERT model. Once again, the custom model provided extends the `BertForSequenceClassification` model of [HuggingFace](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) to integrate the class weights in the loss function.
## Inference
The inferrer only supports single input inference. It handles all the processing steps required to feed the text into the classification model.
It can be used in the following way:
```
model_infer = ModelInferer(config=config, checkpoint_path=checkpoint_path, quantize=True)
model_infer.predict("I hate you from more than you can imagine")
```
We also provide a Flask API that encapsulates the inferrer as well as a way Dockerized the app for production usage.
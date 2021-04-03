import argparse
import logging
import datasets
import os

import neptune
import torch
from dotenv import load_dotenv

from eval import evaluate
from train import train
from utils import config, device, get_featurized_dataset, features_loader_conference, features_loader_toxicity


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--resume-training", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--do-eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model", default="bert-multi-label", type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--data-dir", default="data", type=str)
    parser.add_argument("--get-mismatched", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model-type", type=str, default=config["model_type"])
    parser.add_argument("--model-name", type=str, default=config["model_name"])
    parser.add_argument("--tokenizer-name", type=str, default=config["tokenizer_name"])
    parser.add_argument("--task-name", type=str, default=config["task_name"])
    parser.add_argument("--neptune-username", type=str, default="julesbelveze")
    parser.add_argument("--neptune-project", type=str, default="multi-label-classifier")
    parser.add_argument("--neptune-id", type=str, default=None)
    parser.add_argument("--tags", nargs='+', default=[])
    return parser.parse_args()


def run(args):
    """"""
    # setting up neptune experiment
    neptune_project = neptune.init(
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        project_qualified_name='{}/{}'.format(args["neptune_username"], args["neptune_project"])
    )

    # updating config with the provided flags
    config.update(args)
    logging.info("Used config: {}".format(config))

    model_config = config.model_config
    model_class = config.model_class
    tokenizer_class = config.tokenizer_class

    model_config = model_config.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        finetuning_task=config["task_name"]
    )
    model_config.update(config)

    tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])
    model = model_class(model_config).to(device)

    # resume training
    if config["model_path"]:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))

    if config["task_name"] == "multi-label":
        dataset = datasets.load_dataset("data/toxic_dataset.py")
        dataset = features_loader_toxicity(dataset, tokenizer, max_length=config["max_length"])
    elif config["task_name"] == "multi-class":
        dataset = datasets.load_dataset("data/conference_dataset.py")
        dataset = features_loader_conference(dataset, tokenizer, max_length=config["max_length"])
    else:
        raise ValueError(f"Task name '{config['task_name']}' not supported")
    train_data, test_data = dataset["train"], dataset["test"]
    train_dataset, test_dataset = get_featurized_dataset(tokenizer, train_data, test_data)

    if config["do_train"]:
        global_step, tr_loss = train(train_dataset, test_dataset, model, config, neptune_project)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if config["do_eval"]:
        report = evaluate(test_dataset, model, config)
        logging.info("---------------------- Evaluation report ----------------------\n{}".format(report))


if __name__ == "__main__":
    load_dotenv()
    flags = vars(parse_flags())
    run(flags)

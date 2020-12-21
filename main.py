import argparse
import logging

import torch

from utils import config, Dataset, processors
from eval import evaluate
from train import train


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-train", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--do-eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model", default="bert-multi-label", type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--get-mismatched", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model-type", type=str, default=config["model_type"])
    parser.add_argument("--model-name", type=str, default=config["model_name"])
    parser.add_argument("--tokenizer-name", type=str, default=config["tokenizer_name"])
    parser.add_argument("--task-name", type=str, default=config["task_name"])
    return parser.parse_args()


def run(args):
    """"""
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

    tokenizer = tokenizer_class.from_pretrained(config["tokenizer_name"])
    model = model_class(model_config).to(config["device"])

    processor = processors[config["task_name"]]

    # resume training
    if config["model_path"]:
        model.load_state_dict(torch.load(config["model_path"], map_location=config["device"]))

    # creating dataset object
    dataset = Dataset(task=config["task_name"], tokenizer=tokenizer, processor=processor, labels=config["labels"],
                      truncate_mode=config["truncate_mode"])

    # retrieving test data
    test_dataset = dataset.load_and_cache_examples(train=False, **config)

    if config["do_train"]:
        train_dataset = dataset.load_and_cache_examples(train=True, **config)
        global_step, tr_loss = train(train_dataset, test_dataset, model, processor)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if config["do_eval"]:
        report = evaluate(test_dataset, model, processor)


if __name__ == "__main__":
    flags = vars(parse_flags())
    run(flags)

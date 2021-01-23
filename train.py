import logging
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import neptune
from neptunecontrib.api import log_chart
import torch
import seaborn as sns
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from eval import evaluate
from utils import device
from utils.metrics import get_accuracy, accuracy_thresh


def train(train_dataset, eval_dataset, model, processor, config, neptune_project, freeze_model=False):
    """
    :param train_dataset: iterator on the training set
    :param eval_dataset: iterator on the test set
    :param model: instance of the model to train
    :param processor: processor object used for evaluation
    :param config: Config
    :param freeze_model: whether or not to freeze BERT
    """
    if config["resume_training"]:
        # retrieving and updating already existing experiment
        exp = neptune_project.get_experiments(id=config["neptune_id"])[0]
    else:
        # creating a neptune experiment
        exp = neptune_project.create_experiment(name="{}_{}".format(config["model_type"], str(datetime.now())),
                                                params=config,
                                                upload_source_files=['*.py', "models/", "utils/"],
                                                tags=[config["model_type"]] + config["tags"])

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['train_batch_size'],
                                  drop_last=True)

    num_training_steps = len(train_dataloader) * config['num_train_epochs'] // config['gradient_accumulation_steps']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = math.ceil(num_training_steps * config['warmup_ratio'])
    config['warmup_steps'] = warmup_steps if config['warmup_steps'] else 0

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=num_training_steps)

    # will freeze all the model parameters except the classification part
    if freeze_model:
        model.freeze_bert_encoder()

    # optimization
    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

    # if running on multiple GPUs
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", config['num_train_epochs'])
    logging.info("  Total train batch size  = %d", config['train_batch_size'])
    logging.info("  Gradient Accumulation steps = %d", config['gradient_accumulation_steps'])
    logging.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, train_acc, logging_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    start_epoch = int(config.get("previous_checkpoint", 0) / len(train_dataloader))
    train_iterator = trange(start_epoch, int(config['num_train_epochs']), desc="Epoch")

    # starting training
    for epoch in train_iterator:
        epoch_losses = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # avoiding to feed already seen batches when resuming training
            if config["resume_training"] and global_step < config.get("previous_checkpoint") + 1:
                global_step += 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            if 'distilbert' not in config['model_type']:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if config['model_type'] in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}
            else:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)

            # handle multi-gpus run
            if config["n_gpu"] > 1:
                loss = loss.mean()
            print("\r%f" % loss, end='')
            epoch_losses.append(loss.item())

            if config['task_name'] == "multi-label":
                with torch.no_grad():
                    logits = logits.sigmoid()
                train_acc += accuracy_thresh(logits, inputs["labels"])
            else:
                train_acc += get_accuracy(logits.detach().cpu().numpy(), batch[3].detach().cpu().numpy())

            # gradient accumulation
            if config['gradient_accumulation_steps'] > 1:
                loss = loss / config['gradient_accumulation_steps']

            # optimization
            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config['max_grad_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config['logging_steps'] > 0 and global_step % config['logging_steps'] == 0:
                    exp.log_metric(log_name='lr', y=scheduler.get_lr()[0], x=global_step)
                    exp.log_metric(log_name='train_loss', y=(tr_loss - logging_loss) / config['logging_steps'],
                                       x=global_step)
                    exp.log_metric(log_name='train_acc', y=train_acc / config['logging_steps'], x=global_step)
                    logging_loss = tr_loss
                    train_acc = 0.0

                if config['save_steps'] > 0 and global_step % config['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(config['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logging.info("Saving model checkpoint to %s", output_dir)
                    exp.log_artifact(os.path.join(output_dir, "pytorch_model.bin"),
                                         "pytorch_model_{}.bin".format(global_step))

        # Log metrics
        if config['evaluate_during_training']:
            results = evaluate(eval_dataset, model, processor, config, epoch)
            for key, value in results["scalars"].items():
                exp.log_metric(log_name='eval_{}'.format(key), y=value, x=epoch)

            if "labels_probs" in results["arrays"].keys():
                labels_probs = results["arrays"]["labels_probs"]
                for i in range(labels_probs.shape[0]):
                    fig = plt.figure(figsize=(15, 15))
                    sns.distplot(labels_probs[i], kde=False, bins=100)
                    plt.title("Probability boxplot for label {}".format(i))
                    log_chart(name="dist_label_{}_epoch_{}".format(i, epoch), chart=fig, experiment=exp)
                    plt.close("all")

    return global_step, tr_loss / global_step

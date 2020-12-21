import logging
import math
import os
import neptune
from datetime import datetime

import torch
from eval import evaluate
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.metrics import get_accuracy, accuracy_thresh


def train(train_dataset, eval_dataset, model, processor, config, freeze_model=False):
    neptune.create_experiment(name=str(datetime.now()), params=config, upload_source_files=['*.py'])

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

    if freeze_model:  # will freeze all the model parameters except the classification part
        model.freeze_bert_encoder()

    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

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
    train_iterator = trange(int(config['num_train_epochs']), desc="Epoch")

    for epoch in train_iterator:
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(config['device']) for t in batch)
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

            if config["n_gpu"] > 1:
                loss = loss.mean()
            print("\r%f" % loss, end='')

            if config['task_name'] == "multi-label":
                train_acc += accuracy_thresh(logits, inputs["labels"])
            else:
                train_acc += get_accuracy(logits.detach().cpu().numpy(), batch[3].detach().cpu().numpy())

            if config['gradient_accumulation_steps'] > 1:
                loss = loss / config['gradient_accumulation_steps']

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
                    neptune.log_metric(name='lr', y=scheduler.get_lr()[0], x=global_step)
                    neptune.log_metric(name='train_loss', y=(tr_loss - logging_loss) / config['logging_steps'],
                                       x=global_step)
                    neptune.log_metric(name='train_acc', y=train_acc / config['logging_steps'], x=global_step)
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

        # Log metrics
        if config['evaluate_during_training']:
            results = evaluate(eval_dataset, model, processor, config, epoch)
            for key, value in results["scalars"].items():
                neptune.log_metric.add_scalar(name='eval_{}'.format(key), y=value, x=epoch)

    return global_step, tr_loss / global_step

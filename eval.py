import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from utils import device
from utils.metrics import get_eval_report, get_mismatched, get_multi_label_report


def evaluate(eval_dataset, model, processor, config, epoch=None, prefix=""):
    eval_output_dir = config['output_dir']
    results = {}

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config['eval_batch_size'],
                                 drop_last=True)

    # Evaluation
    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", config['eval_batch_size'])

    eval_loss = 0.0
    preds, target_label = None, None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
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
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            target_label = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            target_label = np.append(target_label, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss /= len(eval_dataloader)
    if config['output_mode'] == "classification":
        probs = F.softmax(torch.Tensor(preds), dim=0).numpy()
        preds = np.argmax(preds, axis=1)
        result = get_eval_report(preds, probs, target_label, eval_loss)
    elif config['output_mode'] == "multi-label-classification":
        result = get_multi_label_report(target_label, preds)
    else:
        raise ValueError(
            "'output_mode' should be either set to 'classification' or 'multi-label-classification' but got {}.".format(
                config["output_mode"])
        )
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results_epoch_{}.txt".format(epoch))
    with open(output_eval_file, "w") as writer:
        logging.info("***** Eval results {} *****".format(prefix))
        display_infos = {key: val for key, val in result.items() if key != 'labels_probs'}
        for key in sorted(display_infos.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if config['get_mismatched']:
        get_mismatched(target_label, preds, processor, config)

    return results

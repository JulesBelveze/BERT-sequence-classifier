import torch
import numpy as np
from pickle import dump
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, auc, \
    roc_curve
from utils import config


def accuracy_thresh(logits, y_true, thresh: float = 0.5):
    """"""
    return torch.mean(((logits > thresh) == y_true).float())


def hamming_loss(logits, y_true, thresh: float = 0.5):
    '''hamming loss: fraction of labels that are incorrectly predicted'''
    preds = (logits > thresh)
    return (preds != y_true).mean()


def get_roc_auc(logits, y_true):
    """"""
    fpr, tpr, roc_auc = {}, {}, {}
    # ROC for each class
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def subset_accuracy(logits, y_true, thresh: float = 0.5):
    '''percentage of samples that hav


    l their labels classified correctly'''
    preds = (logits > thresh)
    return accuracy_score(y_true, preds)


def get_accuracy(preds, labels):
    """"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_multi_label_report(targets, logits, flatten_output=False, thresh: float = 0.5):
    """"""
    fpr, tpr, roc_auc = get_roc_auc(logits, targets)
    hamming = hamming_loss(logits, targets, thresh=thresh)
    acc = accuracy_thresh(torch.tensor(logits), torch.tensor(targets), thresh=thresh)
    subset_acc = subset_accuracy(logits, targets, thresh=thresh)

    preds = (logits > thresh)
    report = classification_report(targets, preds)
    return {
        "scalars": {"auc_micro": roc_auc["micro"], "acc": acc, "subset_acc": subset_acc, "hamming": hamming},
        "dict": {"report": report},
        "arrays": {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
    } if not flatten_output else {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "acc": acc,
        "subset_acc": subset_acc,
        "hamming": hamming,
        "report": report
    }


def get_eval_report(preds, probs, targets, loss_eval, flatten_output=False):
    """"""
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='micro')
    prec = precision_score(targets, preds, average='micro')
    rec = recall_score(targets, preds, average='micro')

    labels_probs = np.array([probs[:, i] for i in range(len(labels))])

    report = classification_report(targets, preds)
    return {
        "scalars": {"f1": f1, "acc": acc, "prec": prec, "rec": rec, "loss_eval": loss_eval},
        "dict": {"report": report},
        "arrays": {"labels_probs": labels_probs}
    } if not flatten_output else {
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "acc": acc,
        "loss": loss_eval,
        "labels_probs": labels_probs,
        "report": report
    }


def get_mismatched(labels, preds, processor, output_mode, save=True):
    """"""
    if output_mode == "multi-label-classification":
        mismatched = (labels > 0.5) != preds
        processor = processor(labels, config["truncate_mode"])
    else:
        mismatched = labels != preds
        processor = processor(labels, config["truncate_mode"])

    examples = processor.get_test_examples(config['data_dir'])

    wrong = [(i, y, y_hat) for (i, v, y, y_hat) in zip(examples, mismatched, labels, preds) if v.any()]

    if save:
        with open("mismatched.pkl", 'wb') as f:
            dump(wrong, f)
    return wrong

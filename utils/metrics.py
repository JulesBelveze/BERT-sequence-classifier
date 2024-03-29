from pickle import dump

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, auc, \
    roc_curve


def accuracy_thresh(probs, y_true, thresh: float = 0.5):
    """
    :param probs: tensor of probabilities
    :param y_true: tensor of ground truth label
    :param thresh: probability threshold for assigning class 1
    :return: accuracy
    """
    return torch.mean(((probs > thresh) == y_true).float())


def hamming_loss(probs, y_true, thresh: float = 0.5):
    """
    Hamming loss: fraction of labels that are incorrectly predicted
    https://en.wikipedia.org/wiki/Multi-label_classification
    :param probs: tensor of probabilities
    :param y_true: tensor of ground truth label
    :param thresh: probability threshold for assigning class 1
    :return : hamming loss
    """
    preds = (probs > thresh)
    return (preds != y_true).mean()


def get_roc_auc(probs, y_true):
    """
    :param probs: tensor of probabilities
    :param y_true: tensor of ground truth label
    :return: fpr, tpr, roc_auc
    """
    fpr, tpr, roc_auc = {}, {}, {}
    # ROC for each class
    for i in range(probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def subset_accuracy(probs, y_true, thresh: float = 0.5):
    """Percentage of samples that have their labels classified correctly
    :param probs: tensor of probabilities
    :param y_true: tensor of ground truth label
    :param thresh: probability threshold for assigning class 1
    """
    preds = (probs > thresh)
    return accuracy_score(y_true, preds)


def get_accuracy(preds, labels):
    """"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_multi_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    Multi metrics computation for multi-label classification problems.
    :param preds: tensor of predictions
    :param labels: tensor of ground truth
    :return:
    """
    tp = (labels * preds).sum().to(torch.float32)
    tn = ((1 - labels) * (1 - preds)).sum().to(torch.float32)
    fp = ((1 - labels) * preds).sum().to(torch.float32)
    fn = (labels * (1 - preds)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1, tp, tn, fp, fn, precision, recall


def get_multi_label_report(targets, probs, flatten_output=False, thresh: float = 0.5):
    """

    :param targets: tensor of ground truth labels
    :param probs: tensor of probabilities
    :param flatten_output: whether or not to return a dict with variable types as keys
    :param thresh: probability threshold to assign class 1
    :return:
    """
    fpr, tpr, roc_auc = get_roc_auc(probs, targets)
    hamming = hamming_loss(probs, targets, thresh=thresh)
    acc = accuracy_thresh(torch.tensor(probs), torch.tensor(targets), thresh=thresh)
    subset_acc = subset_accuracy(probs, targets, thresh=thresh)

    preds = (probs > thresh)
    labels_probs = np.array([probs[:, i] for i in range(probs.shape[1])])
    report = classification_report(targets, preds)
    f1, tp, tn, fp, fn, prec, rec = get_multi_metrics(torch.FloatTensor(preds), torch.FloatTensor(targets))
    return {
        "scalars": {"auc_micro": roc_auc["micro"], "acc": acc, "subset_acc": subset_acc, "hamming": hamming, "f1": f1,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn, "prec": prec, "rec": rec},
        "dict": {"report": report},
        "arrays": {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "labels_probs": labels_probs}
    } if not flatten_output else {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "acc": acc,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prec": prec,
        "rec": rec,
        "subset_acc": subset_acc,
        "hamming": hamming,
        "report": report
    }


def get_eval_report(preds, probs, targets, loss_eval, config, flatten_output=False):
    """"""
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)

    labels_probs = np.array([probs[:, i] for i in range(config["num_labels"])])

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


def get_mismatched(labels, preds, examples, config, save=True, thresh=0.5):
    """
    Function to save mislabeled observations.
    :param labels:
    :param preds:
    :param processor:
    :param config:
    :param save:
    :param thresh:
    :return:
    """
    if config["output_mode"] == "multi-label-classification":
        mismatched = (labels > thresh) != preds
    else:
        mismatched = labels != preds

    wrong = [(i, y, y_hat) for (i, v, y, y_hat) in zip(examples, mismatched, labels, preds) if v.any()]

    if save:
        with open("mismatched.pkl", 'wb') as f:
            dump(wrong, f)
    return wrong

import numpy as np
import os
import torch
import wandb
import csv
from loguru import logger
import matplotlib.pyplot as plt
from numpy import linalg as LA


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def sanity_check(val_train_loader, static_train_loader, train_loader_cls):
    assert (
        val_train_loader.dataset.train_label != static_train_loader.dataset.train_label
    ).sum() == 0  # check gt are the same
    assert (
        val_train_loader.dataset.train_label != train_loader_cls.dataset.train_label
    ).sum() == 0  # check gt are the same
    assert (
        val_train_loader.dataset.train_img != static_train_loader.dataset.train_img
    ).sum() == 0  # check train img are the same
    assert (
        val_train_loader.dataset.train_img != train_loader_cls.dataset.train_img
    ).sum() == 0  # check train img are the same


def get_word_vec(model, pathology):
    tmp_list = []
    for i in pathology:
        current = model.get_embeds(i)
        tmp_list.append(current)
    saved_embd = torch.stack(tmp_list).numpy()
    return saved_embd


def load_word_vec(args, norm=True):
    fn = f"{args.train_data.lower()}_{args.bert_name}_{args.num_classes}.npy"
    logger.bind(stage="CONFIG").critical(f"Loading word embedding {fn}")
    wordvec_array = np.load(f"./embeddings/{args.bert_name}/{fn}")

    if norm:
        normed_wordvec = np.stack(
            [
                wordvec_array[i] / LA.norm(wordvec_array[i])
                for i in range(wordvec_array.shape[0])
            ]
        )
        normed_wordvec = (
            (torch.tensor(normed_wordvec).to(args.device).float())
            .permute(1, 0)
            .unsqueeze(0)
        )
    return normed_wordvec


def print_auroc(aurocIndividual):
    aurocIndividual = np.round(aurocIndividual, 4)

    MeanAUC_15c = np.round(aurocIndividual[0:].mean(), 4)
    MeanAUC_14c = np.round(aurocIndividual[1:].mean(), 4)

    row_csv = list(aurocIndividual).copy()
    row_csv.extend([MeanAUC_15c, MeanAUC_14c])
    # self.df_result.iloc[epoch] = list(aurocIndividual).extend([MeanAUC_15c, MeanAUC_14c])

    f = open(os.path.join(wandb.run.dir, "pred.csv"), "a")
    writer = csv.writer(f)
    writer.writerow(row_csv)
    f.close

    wandb.log({"MeanAUC_14c": MeanAUC_14c})
    logger.bind(stage="TEST").success(
        "| MeanAUC_15c: {} | MeanAUC_14c: {} |".format(MeanAUC_15c, MeanAUC_14c)
    )
    return MeanAUC_14c


def save_checkpoint(state_dict, prefix="best"):
    path = os.path.join(wandb.run.dir, "model_{}.pth".format(prefix))
    torch.save(state_dict, path)
    logger.bind(stage="EVAL").debug(f"Saving {prefix} checkpoint")
    return


def log_csv(epoch, all_auc, mean_auc, mean_map, path):
    array = np.asarray(all_auc.copy()) * 100

    tmp_row = array.tolist()
    tmp_row.extend([mean_auc * 100, mean_map * 100])
    tmp_row.insert(0, [epoch])
    f = open(os.path.join(wandb.run.dir, f"{path}.csv"), "a")
    # f = open(path, "a")
    writer = csv.writer(f)
    writer.writerow(tmp_row)
    f.close


def plot_dist(x, ymax=70000, ystep=5000, color="", name=""):
    plt.bar(np.linspace(1, 14, 14), x, color=color)
    plt.yticks(np.arange(0, ymax, ystep))
    plt.title(f"{name}", fontsize=32)
    plt.savefig(os.path.join(wandb.run.dir, f"./{name}.png"))
    plt.clf()


def upload_dist(matrix, title=""):
    values = matrix.sum(0).astype(int)
    labels = np.linspace(1, 14, 14)
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    wandb.log({f"{title}": wandb.plot.bar(table, "label", "value", title=title)})

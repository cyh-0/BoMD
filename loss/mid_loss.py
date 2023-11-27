import torch
import torch.nn as nn


class MID_LOSS(nn.Module):
    def __init__(self, beta=0.3, wordvec_array=None, args=None):
        super(MID_LOSS, self).__init__()
        self.eps = 1e-7
        self.wordvec_array = wordvec_array
        self.embed_len = args.embed_len
        self.beta = beta
        self.num_fea = args.num_fea

    def forward(self, x, y):
        batch_size = y.shape[0]
        loss = torch.zeros(batch_size).cuda()
        
        un_flat = x.view(x.shape[0], self.embed_len, -1)
        M = un_flat.shape[2]

        dot_prod_all = [
            torch.sum(
                (un_flat[:, :, i].unsqueeze(2) * self.wordvec_array), dim=1
            ).unsqueeze(2)
            for i in range(M)
        ]

        dot_prod_all = torch.max(torch.cat(dot_prod_all, dim=2), dim=-1)
        dot_prod_all = dot_prod_all.values

        for i in range(0, batch_size):
            dot_prod_pos = dot_prod_all[i, y[i] == 1]  
            dot_prod_neg = dot_prod_all[
                i, (1 - y[i]).bool()
            ]  
            if len(dot_prod_neg) == 0:  
                v = -dot_prod_pos.unsqueeze(1)
            else:
                v = dot_prod_neg.unsqueeze(0) - dot_prod_pos.unsqueeze(1)

            num_pos = dot_prod_pos.shape[0]
            total_var = calc_diversity(self.wordvec_array, y[i])
            if self.num_fea == 1:
                loss[i] = torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)
            else:
                loss[i] = (
                    (1 + total_var) * torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)
                )
                l1_err = var_regularization(un_flat[i])
                loss[i] = 2 * ((1 - self.beta) * (loss[i]) + self.beta * l1_err)

        return loss.mean()


def calc_diversity(wordvec_array, y_i):
    rel_vecs = wordvec_array[:, :, y_i == 1]
    rel_vecs = rel_vecs.squeeze(0)
    if rel_vecs.shape[1] == 1:
        sig = rel_vecs * 0
    else:
        sig = torch.var(rel_vecs, dim=1)

    return sig.sum()


def var_regularization(x_i):
    sig2 = torch.var(x_i, dim=1)
    l1_err = torch.norm(sig2, dim=-1, p=1)
    return l1_err

import torch.nn as nn
import torch.distributions as dist
import torch
import torchvision
import torch.nn.functional as F

# from models.PreResNet import ResNet18
from torchvision.models.densenet import densenet121
# from torchvision.models import resnet18, ResNet18_Weights


class CNN(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.0):
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_classes)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        b = x.shape[0]
        h = x
        h = self.c1(h)
        self.leaky_relu(self.bn1(h))
        # h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        self.leaky_relu(self.bn2(h))
        # h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        self.leaky_relu(self.bn3(h))
        # h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        self.leaky_relu(self.bn4(h))
        # h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        self.leaky_relu(self.bn5(h))
        # h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        self.leaky_relu(self.bn6(h))
        # h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        self.leaky_relu(self.bn7(h))
        # h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        self.leaky_relu(self.bn8(h))
        # h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        self.leaky_relu(self.bn9(h))
        # h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        x = self.l_c1(h)
        return x.view(b, self.n_classes // 2, 2)


class PD_NPC(nn.Module):
    def __init__(self, args, num_classes=14, encoder_size=128, input_chs=[512]) -> None:
        """
        class 0: Normal
        class 1: Cancer
        """
        super(PD_NPC, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = num_classes
        self.relu = nn.ReLU()

        self.q = nn.Linear(num_classes * 2 + num_classes *
                           2, self.latent_dim * 2)

        # self.feat = CNN(num_classes * 2, dropout_rate=0.3)

        # self.feat = densenet121(pretrained=True)
        self.feat = torchvision.models.resnet18(pretrained=True)
        self.feat.fc = nn.Linear(512, self.num_classes * 2)

        # self.encoder_linear = nn.Linear(encoder_size, num_classes)
        self.soft_plus = nn.Softplus(beta=1)

        self.criterion = nn.MultiLabelSoftMarginLoss().cuda()

    def forward(self, img, y_tilde, cls_outputs):
        cls_outputs = torch.sigmoid(cls_outputs)
        cls_outputs[cls_outputs>.5]=1
        cls_outputs[cls_outputs<.5]=0
        cls_outputs = cls_outputs.unsqueeze(-1)

        b = img.shape[0]
        # rand_in = (
        #     torch.randint(0, 2, (y_tilde.shape[0], y_tilde.shape[1]))
        #     # .unsqueeze(-1)
        #     .cuda()
        # )
        # rand_in[rand_in[:, -1] == 1, :-1] = 0
        # rand_in = rand_in.unsqueeze(-1)
        rand_in = torch.cat((1 - cls_outputs, cls_outputs), dim=-1)

        # x -> [B, 14, 2]
        x = self.feat(img).view(b, self.num_classes, 2)
        # cat_input -> [B, 14, 4]
        cat_input = torch.cat([x, rand_in], -1)
        tmp = self.soft_plus(self.q(cat_input.view(b, -1)))
        tmp = tmp.view(b, self.num_classes, 2)

        alpha = tmp + 1.0
        # logit_norm = alpha[:, :, 0]
        # logit_cancer = alpha[:, :, 1]
        # Output z follow alpha
        # z = dist.Beta(logit_cancer, logit_norm).rsample()

        return {
            # "n_pred": n_pred,
            # "q_x": x,
            # "z": z,
            "alpha": alpha,
        }

    def loss(self, pack, knn_gt, rho=5):
        alpha = pack["alpha"]
        # Non-zero label
        if (knn_gt.sum(dim=1) > 0).sum().item() != knn_gt.shape[0]:
            raise ValueError("KNN GT ALL ZERO")

        knn_gt_expd = torch.cat(
            (1 - knn_gt.unsqueeze(-1), knn_gt.unsqueeze(-1)), dim=-1
        )
        alpha_prior = knn_gt_expd * rho + 1

        KL_dir = dist.kl_divergence(
            dist.Dirichlet(alpha),
            dist.Dirichlet(alpha_prior)
        )
        kl_loss_mean = torch.mean(KL_dir)
        return kl_loss_mean

    def inference(self, img, labels, prob):
        b = img.shape[0]

        iter_label = torch.eye(self.num_classes, self.num_classes).cuda()
        iter_label_expd = (
            torch.cat((1 - iter_label.unsqueeze(-1),
                      iter_label.unsqueeze(-1)), dim=-1)
            .unsqueeze(0)
            .repeat(b, 1, 1, 1)
            .cuda()
        )
        iter_label = torch.eye(2, 2).cuda()
        iter_label_expd = iter_label.unsqueeze(0).repeat(b,14,1,1)
        iter_label_expd = iter_label_expd.permute(0, 3, 1, 2)

        x = self.feat(img).view(b, self.num_classes, 2)
        cat_input_tmp = torch.cat(
                [x.unsqueeze(1).repeat(1, 2, 1, 1),
                iter_label_expd], -1
            )
        logit = self.soft_plus(self.q(cat_input_tmp.view(b, cat_input_tmp.shape[1], -1)))
        logit = logit.view(b, 2, 14 ,2)
        logit = logit/logit.sum(1,keepdim=True)
        p_y_bar_x_y_tilde = logit.permute(0,2,3,1)

        prob = torch.sigmoid(prob)
        p_y_expansion = torch.cat((1-prob.unsqueeze(-1),prob.unsqueeze(-1)),dim=-1).unsqueeze(-2).repeat(1, 1, 2, 1)
        p_y_y_tilde = p_y_bar_x_y_tilde * p_y_expansion

        pred = p_y_y_tilde.sum(-2)[:,:,1] * prob

        return pred


    def evid_transform(self, logits, rho):
        alpha = self.soft_plus(self.evid(logits)) + 1.0
        alpha_prior = dist.Dirichlet(alpha)
        return alpha_prior

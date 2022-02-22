import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import argparse

import data_loader
from utils import init_weights
from .tester import Tester


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams.act_thresh = np.linspace(
            self.hparams.act_thresh_params[0],
            self.hparams.act_thresh_params[1],
            self.hparams.act_thresh_params[2])
        self.hparams.tIoU_thresh = np.arange(*self.hparams.tIoU_thresh_params)

        self.net = HAMNet(hparams)

        self.tester = Tester(self.hparams)

    def forward(self, x):
        return self.net(x, include_min=self.hparams.adl_include_min == 'true')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         conflict_handler="resolve")
        parser.add_argument("--model_name", type=str, default="thumos")
        parser.add_argument("--rat", type=int, default=10, help="topk value")
        parser.add_argument("--rat2",
                            type=int,
                            default=None,
                            help="topk value")

        parser.add_argument("--beta", type=float, default=0.8)
        parser.add_argument("--alpha", type=float, default=0.8)
        parser.add_argument("--num_segments", type=int, default=500)
        parser.add_argument("--sampling", type=str, default="random")
        parser.add_argument('--class_thresh', type=float, default=0.2)

        # NOTE: work on anet dataset
        parser.add_argument("--dataset_name",
                            type=str,
                            default="Thumos14reduced")

        parser.add_argument("--scale", type=int, default=1)
        parser.add_argument("--lr", type=float, default=1e-5)

        parser.add_argument("--lm_1", type=float, default=1)
        parser.add_argument("--lm_2", type=float, default=1)

        parser.add_argument("--drop_thres", type=float, default=0.2)
        parser.add_argument("--drop_prob", type=float, default=0.2)
        parser.add_argument('--gamma_oic', type=float, default=0.2)
        parser.add_argument('--adl_include_min', type=str, default='true')
        parser.add_argument("--percent-sup", type=float, default=0.1)

        parser.add_argument("--batch_size", type=float, default=20)

        parser.add_argument("--tIoU_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.75, 0.1])
        parser.add_argument("--act_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.9, 10])
        parser.add_argument("--rand", type=str, default='false')
        parser.add_argument("--max_epochs", type=int, default=110)
        return parser

    # --------------------------------- load data -------------------------------- #
    def train_dataloader(self):
        dataset = data_loader.Dataset(self.hparams,
                                      mode='train',
                                      sampling=self.hparams.sampling)
        if self.logger is not None:
            self.logger.experiment.info(
                f"Total training videos: {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = data_loader.Dataset(self.hparams,
                                      mode='test',
                                      sampling='all')
        if self.logger is not None:
            self.logger.experiment.info(
                f"Total testing videos: {len(dataset)}")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        self.class_dict = dataset.class_dict
        return dataloader

    # ---------------------------------- config ---------------------------------- #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=0.001)
        return optimizer

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        total_loss, tqdm_dict = self._loss_fn(batch)

        self.log_dict(tqdm_dict, prog_bar=False, logger=True)
        return total_loss

    # ----------------------------------- test ----------------------------------- #
    def test_step(self, batch, batch_idx):
        self.tester.eval_one_batch(batch, self, self.class_dict)

    def test_epoch_end(self, outputs):
        mAP = self.tester.final(logger=self.logger.experiment,
                                class_dict=self.class_dict)

        mAP = torch.tensor(mAP).to(self.device)
        self.log("mAP", mAP, prog_bar=True)

        self.tester = Tester(self.hparams)

    # ------------------------------ loss functions ------------------------------ #
    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def _loss_fn(self, batch):
        """ Total loss funtion """
        features, labels, segm, vid_name, _ = batch

        element_logits, atn_supp, atn_drop, element_atn,bg_score,bg_norm = self.net(features)

        element_logits_supp = self._multiply(element_logits, atn_supp) #soft attention

        element_logits_drop = self._multiply(
            element_logits, (atn_drop > 0).type_as(element_logits),
            include_min=True) #hard attention
        element_logits_drop_supp = self._multiply(element_logits,
                                                  atn_drop,
                                                  include_min=True) #semi-soft attention

        # BCL
        loss_1_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=self.hparams.rat,
                                       reduce=None)

        # HAL
        loss_1_drop, _ = self.topkloss(element_logits_drop,
                                       labels,
                                       is_back=True,
                                       rat=self.hparams.rat,
                                       reduce=None)

        # SAL
        loss_2_orig_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=self.hparams.rat,
                                            reduce=None)

        # SSAL
        loss_2_drop_supp, _ = self.topkloss(element_logits_drop_supp,
                                            labels,
                                            is_back=False,
                                            rat=self.hparams.rat,
                                            reduce=None)

        wt = self.hparams.drop_prob

        loss_1 = (wt * loss_1_drop + (1 - wt) * loss_1_orig).mean()#origin vs hard

        loss_2 = (wt * loss_2_drop_supp + (1 - wt) * loss_2_orig_supp).mean()#semi-soft vs soft

        loss_norm = element_atn.mean()

        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()
        """
        #version 1
        #bg uni-loss
        gamma = 0.2                                       #paramater
        bs= bg_fuse.size(0)
        bg_fuse_e = bg_fuse[:,:,:-1]
        bg_pred = F.softmax(bg_fuse_e.mean(-1))
        bg_loss = torch.sum(torch.norm(bg_pred,p=2,dim=-1))/bs
        #Pseudo label loss
        bce = nn.BCELoss()
        pse_loss = bce(bg_att,att_label)/bs
        #sum
        Bg_loss = 1 * bg_loss + 1.5 * pse_loss   #parameter
        """

        #version 2
        bs = element_logits.size(0)

        Bg_score = torch.max(bg_score, -1)[0].unsqueeze(-1)
        Bg_score_max = torch.max(Bg_score, dim=-2, keepdim=True)[0]
        Bg_score_min = torch.min(Bg_score, dim=-2,keepdim=True)[0]
        Bg_score = (Bg_score-Bg_score_min)/(Bg_score_max-Bg_score_min)
        # Bg_score = (Bg_score >= 1).type_as(element_logits)
        bg_fuse = element_logits * Bg_score
        bg_label = torch.zeros(bs, 20).cuda()
        bg_cls_loss, _ = self.topkloss(bg_fuse, bg_label, is_back=True,
                                       rat=self.hparams.rat,
                                       reduce=None)
        bg_cls_loss = bg_cls_loss.mean()

        bg_thresh = 0.5
        bg_score = (bg_score > bg_thresh).type_as(element_logits)  # (bs,t,l)
        bg_p = torch.sum(bg_score, -1)
        bg_in = torch.zeros_like(bg_p)
        bg_out = torch.where(bg_p > 1,(bg_p-1),bg_in)
        # bg_uniloss = torch.norm(bg_out,p=2,dim=-1).mean()

        M = torch.matmul(bg_norm,bg_norm.transpose(-1,-2))
        m = M.size(-1)
        bg_uniloss = torch.norm(M-torch.eye(m).cuda(), p='fro')

        gamma = 0.1
        bg_loss = 0* bg_uniloss + 1 * bg_cls_loss
        # total loss
        total_loss = (self.hparams.lm_1 * loss_1 + self.hparams.lm_2 * loss_2 +
                      self.hparams.alpha * loss_norm +
                      self.hparams.beta * loss_guide+ gamma * bg_loss)
        tqdm_dict = {
            "loss_train": total_loss,
            "loss_1": loss_1,
            "loss_2": loss_2,
            "loss_norm": loss_norm,
            "loss_guide": loss_guide,
            "bg_uniloss": bg_uniloss,
            "bg_cls_loss": bg_cls_loss

        }

        return total_loss, tqdm_dict

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )

        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))

        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind


# ---------------------------------------------------------------------------- #
#                                     model                                    #
# ---------------------------------------------------------------------------- #
class HAMNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_class = args.num_class
        n_feature = args.feature_size

        self.classifier = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(n_feature, n_class + 1, 1))

        self.attention = nn.Sequential(nn.Conv1d(n_feature, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Sigmoid())
        self.adl = ADL(drop_thres=args.drop_thres, drop_prob=args.drop_prob)
        self.apply(init_weights)

        #background fine-grained feature
        self.L = 3          #bg number
        self.cos_factor = 5 #cossim factor
        self.T =2.0 #attention temprature
        self.bg_cls = nn.Parameter(torch.zeros(self.L,n_feature))
        self.bg_thresh = 0.5
        torch.nn.init.kaiming_uniform(self.bg_cls)

    def forward(self, inputs, include_min=False):
        x = inputs.transpose(-1, -2) #(bs,n_feature,n_segment)
        x_cls = self.classifier(x)
        x_atn = self.attention(x)
        atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=include_min)

        #new into input feature
        bg_norm = self.cal_l1_norm(self.bg_cls)
        inputs_norm = self.cal_l1_norm(inputs)
        bg_score = torch.einsum('ntd,ld->ntl',[inputs_norm,bg_norm])
        bg_score = F.softmax(bg_score,dim=-1)
        #add contrastive learning
        #delete contrastive learning
        
        # bg_score_max = torch.max(bg_score,dim=-2,keepdim=True)[0]
        # bg_score_min = torch.min(bg_score,dim=-2,keepdim=True)[0]
        # bg_score = (bg_score-bg_score_min)/(bg_score_max-bg_score_min)

        # bg_score = (bg_score > self.bg_thresh).type_as(x)#(bs,t,l)

        """
        bg_atts = [F.softmax(bg_score*t,dim=-1) for t in self.T]
        bg_feas = [torch.einsum('ntd,ntl->nld',[inputs,bg_att]) for bg_att in bg_atts]
        bg_xs= [bg_fea.transpose(-1,-2) for bg_fea in bg_feas]
        bg_score_fuse = [self.classifier(bg_x) for bg_x in bg_xs ]
        bg_fuse = torch.stack(bg_score_fuse,-1).mean(-1)
        """
        """
        #new into attention feature
        bg_at = torch.max(bg_score,-1)[0]
        bg_att_max = torch.max(bg_at,-1,keepdim=True)[0]
        bg_att_min = torch.min(bg_at,-1,keepdim=True)[0]
        bg_Att = (bg_at-bg_att_min)/(bg_att_max-bg_att_min)

        atn_max = x_atn.max(dim=-1, keepdim=True)[0]
        atn_min = x_atn.min(dim=-1, keepdim=True)[0]
        label_factor = 0.4                            #parameter
        label_thres = (atn_max - atn_min) * label_factor + atn_min
        att_label = 1 - (x_atn > label_thres).type_as(x).squeeze(1)
        """
        return x_cls.transpose(-1, -2), atn_supp.transpose(
            -1, -2), atn_drop.transpose(-1, -2), x_atn.transpose(-1, -2),\
             bg_score,bg_norm

    def cal_l1_norm(self,x):
        x_norm = torch.norm(x,p=2,dim=-1,keepdim=True)
        x_out = torch.div(x,x_norm)
        return x_out

class ADL(nn.Module):
    def __init__(self, drop_thres=0.5, drop_prob=0.5):
        super().__init__()
        self.drop_thres = drop_thres
        self.drop_prob = drop_prob

    def forward(self, x, x_atn, include_min=False):
        if not self.training:
            return x_atn, x_atn

        # important mask
        mask_imp = x_atn

        # drop mask
        if include_min:
            atn_max = x_atn.max(dim=-1, keepdim=True)[0]
            atn_min = x_atn.min(dim=-1, keepdim=True)[0]
            _thres = (atn_max - atn_min) * self.drop_thres + atn_min
        else:
            _thres = x_atn.max(dim=-1, keepdim=True)[0] * self.drop_thres
        drop_mask = (x_atn < _thres).type_as(x) * x_atn

        return mask_imp, drop_mask

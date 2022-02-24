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

        parser.add_argument("--batch_size", type=float, default=100)

        parser.add_argument("--tIoU_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.75, 0.1])
        parser.add_argument("--act_thresh_params",
                            type=float,
                            nargs=3,
                            default=[0.1, 0.9, 10])
        parser.add_argument("--rand", type=str, default='false')
        parser.add_argument("--max_epochs", type=int, default=100)
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

    def val_dataloader(self):
        return self.test_dataloader()

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

    # --------------------------------- validation --------------------------------- #
    def validation_step(self,batch, batch_idx):
        self.tester.eval_one_batch(batch, self, self.class_dict)

    def validation_epoch_end(self, outputs):
        mAP = self.tester.final(logger=self.logger.experiment,
                                class_dict=self.class_dict)
        mAP = torch.tensor(mAP).to(self.device)
        self.log("mAP", mAP, prog_bar=True)

        self.tester = Tester(self.hparams)
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

        element_logits, atn_supp, atn_drop, element_atn,\
        bg_cls_agg, bg_fea_norm, contr_ele, _ = self.net(features)

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

        #bg cls loss
        bs = bg_cls_agg.size(0)
        # bg_cls_loss = torch.sum(torch.norm(bg_cls_agg,p=2,dim=-1))/bs
        bg_cls_loss = -torch.sum(torch.log(bg_cls_agg))/bs
        #diversity loss
        M = torch.matmul(bg_fea_norm, bg_fea_norm.transpose(-1, -2))
        m = M.size(-1)
        bg_div_loss = torch.norm(M - torch.eye(m).cuda(), p='fro')

        #contrastive loss
        bgref_norm = contr_ele[0] #(l,d)
        fg_norm = contr_ele[1]    #(n,t,d)
        bg_norm = contr_ele[2]    #(n,t,d)
        T = bg_norm.size(1)

        bg_sim = torch.einsum('ntd,nld->ntl',[bg_norm,bgref_norm])*10
        bg_sim_T, bg_sim_id = torch.max(bg_sim,-1) #(n,t)

        """待修改"""
        bf_sim1 = torch.einsum('ntd,nld->ntl',[fg_norm,bgref_norm])*10 #(n,t,l)
        bg_id = torch.arange(0,bg_sim_T.size(0)).view(-1,1).repeat(1,bg_sim_T.size(1))
        contr_loss1 = -torch.log(torch.exp(bg_sim_T)/(torch.exp(bg_sim_T)+torch.exp(bf_sim1).sum(-2)[bg_id,bg_sim_id]))
        bf_sim2 = torch.einsum('ntd,nsd->nts',[fg_norm,bg_norm])*10 #(n,t,l)
        contr_loss2 = -torch.log(torch.exp(bg_sim_T)/(torch.exp(bg_sim_T)+torch.exp(bf_sim2).sum(-2)))
        contr_loss = (contr_loss1.sum() + contr_loss2.sum())/T

        gamma = 0
        bg_loss = 1 * bg_cls_loss + 1 * bg_div_loss + 0 * contr_loss
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
            "bg_cls_loss": bg_cls_loss,
            "bg_div_loss": bg_div_loss,
            "contr_loss":contr_loss
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
        self.projection = nn.Linear(n_feature, 128)
        self.apply(init_weights)

        #background fine-grained feature
        self.L = 6          #bg number
        self.cos_factor = 5 #cossim factor
        self.T = [5.0] #attention temprature
        self.bg_cls = nn.Parameter(torch.zeros(self.L,n_feature))
        self.bg_thresh = 0.5
        torch.nn.init.kaiming_uniform(self.bg_cls)


    def forward(self, inputs, include_min=False):
        x = inputs.transpose(-1, -2) #(bs,n_feature,n_segment)
        x_cls = self.classifier(x)
        x_atn = self.attention(x) #(n,1,t)
        atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=include_min)

        #bg feature branch
        bg_norm = self.cal_l1_norm(self.bg_cls)
        inputs_norm = self.cal_l1_norm(inputs)
        bg_score = torch.einsum('ntd,ld->ntl',[inputs_norm,bg_norm])*self.cos_factor

        bg_atts = [F.softmax(bg_score * t, dim=-1) for t in self.T]
        bg_feas = [torch.einsum('ntd,ntl->nld', [inputs, bg_att]) for bg_att in bg_atts]
        bg_fea_norms = [self.cal_l1_norm(bg_fea) for bg_fea in bg_feas]
        bg_feas_trans = [bg_fea_norm.transpose(-1, -2) for bg_fea_norm in bg_fea_norms] #(n,d,l)
        bg_cls = [self.classifier(bg_x) for bg_x in bg_feas_trans] #[(n,c+1,l)]
        bg_cls_fuse = torch.stack(bg_cls, -1).mean(-1)  #(n,c+1,l,a)->(n,c+1,l) verified
        bg_cls_agg = torch.max(bg_cls_fuse,-1)[0][:,:-1] #(n,c)
        bg_cls_agg = F.softmax(bg_cls_agg,-1) #(n,c)


        #TEST
        bg_score_ori = torch.max(bg_score,-1)[0]
        bg_score = F.softmax(bg_score,dim=-1)
        bg_score_sf = torch.max(bg_score,-1)[0]
        Bg_score =[bg_score_ori,bg_score_sf]

        #add contrastive learning
        bg_projection_ref = self.projection(bg_feas[0]) #(n,l,d') 参照背景特征
        bg_proref_norm = self.cal_l1_norm(bg_projection_ref)
        fg_id = (x_atn > 0.6).type_as(x_atn).transpose(-1,-2) #(n,t,1)

        fg_projection = self.projection(inputs) #(n,t,d')
        fg_pro_norm = self.cal_l1_norm(fg_projection)
        fg_pro_norm = fg_pro_norm * fg_id
        bg_id = (x_atn < 0.2).type_as(x_atn).transpose(-1,-2)

        bg_projection = self.projection(inputs)
        bg_pro_norm = self.cal_l1_norm(bg_projection)
        bg_pro_norm = bg_pro_norm * bg_id

        contr_ele = [bg_proref_norm,fg_pro_norm,bg_pro_norm]


        return x_cls.transpose(-1, -2), atn_supp.transpose(
            -1, -2), atn_drop.transpose(-1, -2), x_atn.transpose(-1, -2),\
             bg_cls_agg,bg_fea_norms[0],contr_ele,Bg_score

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

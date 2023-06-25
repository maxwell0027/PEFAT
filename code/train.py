import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import torch
from torchmetrics import Specificity
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121
from sklearn.metrics import multilabel_confusion_matrix
from utils import ramps
import torch.nn.functional as F
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from sklearn.mixture import GaussianMixture
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from utils.suploss import SupConLoss
from validation import epochVal, epochVal_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/disk1/mk/ibot-main/data/imagenet/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/media/disk1/mk/ibot-main/data/imagenet/train.txt', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/media/disk1/mk/ibot-main/data/imagenet/val.txt', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/media/disk1/mk/ibot-main/data/imagenet/test.txt', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int,  default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=8, help='number of labeled data per batch')
parser.add_argument('--K', type=int, default=8, help='iteration number')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=100, help='number of labeled')
parser.add_argument('--train_data_num', type=int, default=70000, help='number of training data')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0,1,2,3', help='GPU to use')
parser.add_argument('--warmup_epoch', type=int,  default=30, help='warmup epoch')
### N/A
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### N/A
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        

def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x  


def sensitivityCalc(Predictions, Labels, micro=None, macro=None):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)

    tn_sum = MCM[:, 0, 0] # True Negative
    fp_sum = MCM[:, 0, 1] # False Positive

    tp_sum = MCM[:, 1, 1] # True Positive
    fn_sum = MCM[:, 1, 0] # False Negative

    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
    macro_sensitivity = np.average(sensitivity, weights=None)

    micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum+fn_sum)
    
    if micro != None:
        return micro_sensitivity
    else:
        return macro_sensitivity

def specificityCalc(Predictions, Labels, micro=None, macro=None):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]

    tp_sum = MCM[:, 1, 1]
    fn_sum = MCM[:, 1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    if micro != None:
        return macro_specificity
    else:
        return micro_specificity

def contrastive_loss(x, x_aug, T=0.05):
    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = pos_sim / (sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()
    return loss



if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                 betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer_warmup = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999), weight_decay=5e-4)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                transforms.RandomRotation(10),
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])))
                                            
    eval_train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ])), eval_train=True)


    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.train_data_num))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    eval_train_dataloader = DataLoader(dataset=eval_train_dataset, batch_sampler=batch_sampler,
                                  shuffle=False, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=False)#, worker_init_fn=worker_init_fn)
    
    model.train()


    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()
    
    celoss = torch.nn.CrossEntropyLoss(reduction='mean')


    def mixup_data(x, y, alpha=32.0, device='cuda'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    
        batch_size = x.size()[0]
        if device=='cuda':
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
    
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
        
    def mixup_criterion(pred, y_a, y_b, lam):    
        return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)
        
    CE = torch.nn.CrossEntropyLoss(reduction='none')
    SupLoss = SupConLoss()

    
    logging.info('----------------------stage one: warm up------------------------------------------')
    for warm_up in range(args.warmup_epoch):
        loss_each = []
        acc_each = []
        
        for i, (_,_, (image1, image2), label) in enumerate(train_dataloader):      # weak and base augmentation
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()
            target = torch.argmax(label, dim=1)
            
            feat1, output1 = model(image1)
            feat2, output2 = model(image2)
            
            output_softmax1 = torch.softmax(output1, dim=1)
            output_softmax2 = torch.softmax(output2, dim=1)
            
            # ce for labeled data and contrastive_loss for all training data
            los_ce = celoss(output1[:labeled_bs], target[:labeled_bs]) + celoss(output2[:labeled_bs], target[:labeled_bs])    #[:labeled_bs]
            los_nce = contrastive_loss(feat1, feat2)
           
            loss = los_ce + los_nce
            
            #print('los_ce:{}, los_nce:{}'.format(los_ce, los_nce))
            
            optimizer_warmup.zero_grad()
            loss.backward()
            optimizer_warmup.step()
            
        print('save warm upped model for epoch{}'.format(str(int(warm_up)+1)))    
        save_mode_path = os.path.join(snapshot_path, 'warmup_'+str(int(warm_up)+1)+'.pth')
        torch.save(model.state_dict(), save_mode_path)
        

    #checkpoint = torch.load('/media/disk1/qjzeng/PEFAT/model/xxxx/warmup_best.pth')
    #model.load_state_dict(checkpoint)
    

    def gmm_fit(model):
        with torch.no_grad():
            all_losses = torch.tensor([]).cuda()
            u_losses = torch.tensor([]).cuda()
            for i, (_,_, image1, image2, label) in enumerate(eval_train_dataloader):
                image1, image2, label = image1[:labeled_bs].cuda(), image2[:labeled_bs].cuda(), label[:labeled_bs].cuda()
                image = torch.cat((image1, image2), dim=0)
                label = torch.cat((label, label), dim=0)
                targets = torch.argmax(label, dim=1)
                _, outputs = model(image)
                loss_cls_sup = CE(outputs, targets)
                all_losses = torch.cat((all_losses, loss_cls_sup), -1)
               
            
            all_losses = all_losses.view(-1)
            all_losses = (all_losses-all_losses.min())/(all_losses.max()-all_losses.min())
            all_losses = all_losses.view(-1,1)    
            ema_gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
            ema_gmm.fit(all_losses.cpu())
            
        return ema_gmm


    logging.info('----------------------stage two: loss distribution modeling and feature adversarial training------------------------------------------')
    
    # lose model
    with torch.no_grad():
        all_losses = torch.tensor([]).cuda()
        u_losses = torch.tensor([]).cuda()
        for i, (_,_, image1, image2, label) in enumerate(eval_train_dataloader):
            image1, image2, label = image1[:labeled_bs].cuda(), image2[:labeled_bs].cuda(), label[:labeled_bs].cuda()
            image = torch.cat((image1, image2), dim=0)
            label = torch.cat((label, label), dim=0)
            targets = torch.argmax(label, dim=1)
            _, outputs = model(image)
            loss_cls_sup = CE(outputs, targets)
            all_losses = torch.cat((all_losses, loss_cls_sup), -1)

           
        
        all_losses = all_losses.view(-1)
        all_losses = (all_losses-all_losses.min())/(all_losses.max()-all_losses.min())
        all_losses = all_losses.view(-1,1)
    

    
    ema_gmm = GaussianMixture(n_components=2,max_iter=50,tol=1e-2,reg_covar=5e-4)
    ema_gmm.fit(all_losses.cpu())
    
    


    #train
    for epoch in range(args.start_epoch, args.epochs):
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        
        
        iter_max = len(train_dataloader)    
        for i, (_,_, (image1, image2), label) in enumerate(train_dataloader):
            image1, image2, label = image1.cuda(), image2.cuda(), label.cuda()
            lab_img1, lab_img2, un_img1, un_img2 = image1[:labeled_bs], image2[:labeled_bs], image1[labeled_bs:], image2[labeled_bs:]
            lab_label, un_label = label[:labeled_bs], label[labeled_bs:]
            lab_target, un_target = torch.argmax(lab_label, dim=1), torch.argmax(un_label, dim=1)
            
            
            # designed for unlabeled data
            with torch.no_grad():
                _, un_output1 = model(un_img1)
                _, un_output2 = model(un_img2)
                un_output1_softmax = F.softmax(un_output1, dim=1)
                un_output2_softmax = F.softmax(un_output2, dim=1)
                
                un_prob1, un_pred1 = torch.max(un_output1_softmax, dim=1)
                un_prob2, un_pred2 = torch.max(un_output2_softmax, dim=1)
                un_prob, un_pred = torch.max(0.5*(un_output1_softmax+un_output2_softmax), dim=1)
                un_prob1 = un_prob1.cpu().numpy()
                un_prob2 = un_prob2.cpu().numpy()
                un_prob = un_prob.cpu().numpy()
                
                los_12 = CE(un_output2, un_pred1)
                los_21 = CE(un_output1, un_pred2)
                
                b_size = un_img1.shape[0]
                loss_tmp12 = torch.zeros(b_size)
                loss_tmp21 = torch.zeros(b_size)

                
                for r in range(b_size):
                    loss_tmp12[r] = los_12[r]
                    loss_tmp21[r] = los_21[r]
                    
                loss_tmp12 = (loss_tmp12-loss_tmp12.min())/(loss_tmp12.max()-loss_tmp12.min())
                loss_tmp21 = (loss_tmp21-loss_tmp21.min())/(loss_tmp21.max()-loss_tmp21.min())
                

                loss_tmp12 = loss_tmp12.view(-1,1)
                loss_tmp21 = loss_tmp21.view(-1,1)


                prob12 = ema_gmm.predict_proba(loss_tmp12) 
                prob12 = prob12[:,ema_gmm.means_.argmin()]    
                
                prob21 = ema_gmm.predict_proba(loss_tmp21)
                prob21 = prob21[:,ema_gmm.means_.argmin()]    
                 
                
                prob_comb = ema_gmm.predict_proba(0.5*loss_tmp12+0.5*loss_tmp21)
                prob_comb = prob_comb[:,ema_gmm.means_.argmin()]                            


            trust_idx = prob_comb > 0.70   #0.5
            unc_idx = trust_idx == False

            
            ltru = np.sum(trust_idx)
            lunc = np.sum(unc_idx)
            

            
            
            # divide into two categories based on gmm prediction
            trust_img, trust_lab = torch.cat((lab_img1, un_img1[trust_idx], lab_img2, un_img2[trust_idx]), dim=0), \
                                   torch.cat((lab_target, un_pred2[trust_idx], lab_target, un_pred1[trust_idx]), dim=0)
            #trust_img, trust_lab = torch.cat((lab_img1, lab_img2), dim=0), torch.cat((lab_target, lab_target), dim=0)
            uncer_img1, uncer_img2, mps_lab1, mps_lab2 = un_img1[unc_idx], un_img2[unc_idx], un_pred2[unc_idx], un_pred1[unc_idx]
            
            
            # add perturbation on trust_img and loss for trustworthy images
            # forward once 
            trust_feat, trust_output = model(trust_img)
            t_o_s = torch.softmax(trust_output, dim=1)
            
            #add perturbation on trustworthy images
            d = np.random.normal(size=trust_feat.shape)
            d = normalize_l2(d)
            for iter_num in range(args.K):
                x_d = torch.tensor(trust_feat.clone().detach().cpu().data.numpy() + d.astype(np.float32), requires_grad=True)
                t_d_logit = model.module.densenet121.classifier(x_d.cuda())
                
                cls_loss_td = celoss(t_d_logit, trust_lab)   
                cls_loss_td.backward(retain_graph=True)
                d = x_d.grad
                d = d.numpy()

            trust_feat_at = trust_feat.clone().detach().cpu().data.numpy() + d   
            trust_feat_at = torch.tensor(trust_feat_at).cuda()
            t_outputs_at = model.module.densenet121.classifier(trust_feat_at)
            
            los_t_ce = celoss(trust_output, trust_lab) + 0.3*celoss(t_outputs_at, trust_lab)
            
            
            
            # los for the rest uncertainty image
            feat_u1, unc_output1 = model(uncer_img1)
            feat_u2, unc_output2 = model(uncer_img2)
            unc_o_s1 = torch.softmax(unc_output1, dim=1)
            unc_o_s2 = torch.softmax(unc_output2, dim=1)
            
            du1 = np.random.normal(size=feat_u1.shape)
            du2 = np.random.normal(size=feat_u2.shape)
            for iter_num in range(args.K):
                x_du1 = torch.tensor(feat_u1.clone().detach().cpu().data.numpy() + 1e-3*du1.astype(np.float32), requires_grad=True)
                u_d_logit1 = model.module.densenet121.classifier(x_du1.cuda())
                u_d1_s = torch.softmax(u_d_logit1, dim=1)
                x_du2 = torch.tensor(feat_u2.clone().detach().cpu().data.numpy() + 1e-3*du2.astype(np.float32), requires_grad=True)
                u_d_logit2 = model.module.densenet121.classifier(x_du2.cuda())
                u_d2_s = torch.softmax(u_d_logit2, dim=1)
                
                cls_loss_ud = F.kl_div(u_d2_s.log(), unc_o_s1.detach(), reduction='batchmean') + F.kl_div(u_d1_s.log(), unc_o_s2.detach(), reduction='batchmean')
                cls_loss_ud.backward(retain_graph=True)
                du1 = x_du1.grad
                du2 = x_du2.grad
                du1 = du1.numpy()
                du2 = du2.numpy()
                du1 = normalize_l2(du1)
                du2 = normalize_l2(du2)

            uncer_feat_at1 = feat_u1.clone().detach().cpu().data.numpy() + du1  
            uncer_feat_at2 = feat_u2.clone().detach().cpu().data.numpy() + du2
            uncer_feat_at1 = torch.tensor(uncer_feat_at1).cuda()
            uncer_feat_at2 = torch.tensor(uncer_feat_at2).cuda()
            unc_output_at1 = model.module.densenet121.classifier(uncer_feat_at1)
            unc_output_at2 = model.module.densenet121.classifier(uncer_feat_at2)
            unc_o_ats1 = torch.softmax(unc_output_at1, dim=1)
            unc_o_ats2 = torch.softmax(unc_output_at2, dim=1)
            
            los_unc_at = F.kl_div(unc_o_ats2.log(), unc_o_s1, reduction='batchmean') + F.kl_div(unc_o_ats1.log(), unc_o_s2, reduction='batchmean')
            
            #los_unc_en = -torch.mean(unc_o_ats1*unc_o_ats1.log()) -torch.mean(unc_o_ats2*unc_o_ats2.log()) \
            #             - torch.mean(unc_o_s2*unc_o_s2.log()) - torch.mean(unc_o_s1*unc_o_s1.log()) 
            
            
            
            
            print('los_t_ce:{}, los_unc_at:{}'.format(los_t_ce, los_unc_at))
            
            
            loss = los_t_ce + 0.1*los_unc_at #+ los_unc_en

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1

        
        
        gt = torch.tensor([])
        pred_all = torch.tensor([])
        prob_all = torch.tensor([])            
            
        for i, (_,_, image_batch, label_batch) in enumerate(test_dataloader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            inputs = image_batch
            
            with torch.no_grad():
                target = torch.argmax(label_batch, dim=1)
                activations, outputs = model(inputs)
                output_softmax = F.softmax(outputs, dim=1)
                prob, pred = torch.max(output_softmax, 1)
                
            gt = torch.cat((gt, target.cpu()), 0)
            pred_all = torch.cat((pred_all, pred.cpu()), 0)
            prob_all = torch.cat((prob_all, output_softmax.cpu()), 0)
                
        acc = metrics.accuracy_score(gt, pred_all)
        print('acc: '+str(acc))
        f1 = metrics.f1_score(gt, pred_all, average='macro')
        print('f1: '+str(f1))
        prec = metrics.precision_score(gt, pred_all, average='macro')
        print('prec: '+str(prec))
        auc = metrics.roc_auc_score(gt, prob_all, multi_class='ovo')
        print('auc: '+str(auc))    
        spec = specificityCalc(pred_all, gt, macro=True)
        print('spec: '+str(spec))
        sens = sensitivityCalc(pred_all, gt, macro=True)
        print('sens: '+str(sens))
        print()
        
        
        with open('results.txt', 'a+') as f:
            f.write('acc:{}, f1:{}, prec:{}, auc:{}, spec:{}, sens:{}'.format(acc, f1, prec, auc, spec, sens))
            f.write('\n')

        # update learning rate
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
            

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

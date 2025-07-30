import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from model.model_pose import DCBF
from model.training import Trainer
from model.data_utils import trans_list
from model.optim_init import init_optimizer, init_scheduler
from model.args import create_exp_dirs
from model.args import init_parser, init_sub_args
from model.dataset import get_dataset_and_loader
from model.train_utils import dump_args, init_model_params
from model.scoring_utils import score_dataset
from model.train_utils import calc_num_of_params
import pickle
import copy
from torch.utils.data import ConcatDataset, DataLoader

def main():
    parser = init_parser()
    args = parser.parse_args()
       
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    dataset, loader = get_dataset_and_loader(args, trans_list=None, only_test=True)
    model_args = init_model_params(args, dataset)
    model = DCBF(**model_args)

    trainer = Trainer(args, model, loader['train'], loader['test'],
                        optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                        scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    trainer.load_checkpoint(args.checkpoint)

    normality_scores = trainer.test()
    auc_roc, scores, auc_precision_recall, eer, eer_threshold, gt_np = score_dataset(
        normality_scores, dataset["test"].metadata, args=args)

    print(f"\nDone with {auc_roc*100:.2f}% AuC_ROC, {auc_precision_recall*100:.2f}% Auc_PR, {eer*100:.2f}% EER for {scores.shape[0]} samples\n")

if __name__ == '__main__':
    main()

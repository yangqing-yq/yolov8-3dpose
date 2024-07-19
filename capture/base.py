import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from utils.timer import Timer
from utils.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
# from config import cfg
from model import Model
from data.dataset import MultipleDatasets
import numpy as np

# dynamic dataset import
datasets = ['Fit3D', 'Human36M', 'InterHand26M', 'MPII', 'MSCOCO', 'FreiHAND', 'MPI_INF_3DHP', 'PW3D']
for data in datasets:
    exec('from data.' + data + '.' + data + ' import ' + data)
from data.MSCOCO.MSCOCO_HAND import MSCOCO_HAND

log_dir = 'train_infos/body/logs'


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_dir='train_infos/body/logs', log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(log_dir=log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self, configs):
        super(Trainer, self).__init__(log_dir=osp.join(configs['output_dir'], configs['save_name'], configs['log_dir']), log_name = 'train_logs.txt')
        self.configs = configs

    def get_optimizer(self, model):
        # total_params = []
        # for module in model.module.trainable_modules:
            # total_params += list(module.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'])
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.configs['output_dir'], self.configs['save_name'], self.configs['model_dir'],'snapshot_{}.pth.tar'.format(str(epoch)))

        # do not save human model layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k or 'mano_layer' in k or 'flame_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(self.configs['output_dir'], self.configs['save_name'], self.configs['model_dir'],'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(self.configs['output_dir'], self.configs['save_name'], self.configs['model_dir'], 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in self.configs['lr_dec_epoch']:
            if epoch < e:
                break
        if epoch < self.configs['lr_dec_epoch'][-1]:
            idx = self.configs['lr_dec_epoch'].index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.configs['lr'] / (self.configs['lr_dec_factor'] ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.configs['lr'] / (self.configs['lr_dec_factor'] ** len(self.configs['lr_dec_epoch']))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(self.configs['trainset_3d'])):
            trainset3d_loader.append(eval(self.configs['trainset_3d'][i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(self.configs['trainset_2d'])):
            trainset2d_loader.append(eval(self.configs['trainset_2d'][i])(transforms.ToTensor(), "train"))
       
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.itr_per_epoch = math.ceil(len(trainset_loader) / self.configs['num_gpus'] / self.configs['train_batch_size'])
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.configs['num_gpus']*self.configs['train_batch_size'], shuffle=True, num_workers=self.configs['num_thread'], pin_memory=True, drop_last=True)

    def _make_model(self, pretrained_weight, parts):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = Model(pretrained_weight, 50, 'train', parts)
        model = model.cuda()
        optimizer = self.get_optimizer(model)
        # if cfg.continue_train:
        #     start_epoch, model, optimizer = self.load_model(model, optimizer)
        # else:
        start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def _make_model_new(self, pretrained_weight, parts, continue_train=False):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = Model(pretrained_weight, 50, 'train', parts)
        model = model.cuda()
        optimizer = self.get_optimizer(model)
        if continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self, test_epoch, configs):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_dir=osp.join(configs['output_dir'], configs['save_name'], configs['log_dir']), log_name = 'test_logs.txt')
        self.configs = configs

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(self.configs['testset'])(transforms.ToTensor(), "test")
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.configs['num_gpus']*self.configs['test_batch_size'], shuffle=False, num_workers=self.configs['num_thread'], pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator

    def _make_model(self, pretrained_weight, parts):
        model_path = os.path.join(self.configs['output_dir'], self.configs['save_name'], self.configs['model_dir'], 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = Model(pretrained_weight, 50, 'test', parts)
        model = model.cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _make_model_new(self, pretrained_weight, parts, test_epoch):
        model_path = os.path.join(self.configs['output_dir'], self.configs['save_name'], self.configs['model_dir'],
                                  'snapshot_%d.pth.tar' % test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = Model(pretrained_weight, 50, 'test', parts)
        model = model.cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)
        self.logger.info('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        self.logger.info('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        self.logger.info('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        self.logger.info('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))



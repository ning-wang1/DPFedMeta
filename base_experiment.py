import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import os
import time
import random
import numpy as np

from data_tasks_split import data_loader
from utils.storage import save_statistics

K_FOLDS = 5


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ExperimentBuilder(object):
    def __init__(self, device, args, model, train_data, test_data, lr, batch_size, full_batch=False, pretrain=False):
        self.device = device
        self.use_cuda = args.use_cuda
        self.args = args
        self.model = model
        self.model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.full_batch = full_batch

        self.fold_num = 0
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        if pretrain:
            self.epochs = args.total_epochs
            #self.lr = args.meta_learning_rate
            self.lr = 0.01
        else:
            self.epochs = 30
            self.lr = lr

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.epochs,
                                                              eta_min=args.min_learning_rate)
        self.experiment_path, self.logs_fp = self.build_log_path()
        self.train_sum_name = 'train_summary.csv'
        self.test_sum_name = 'test_summary.csv'

        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.S = 4
        self.noise_multiplier = 6
        self.grad_layer_num = self.get_grad_layer_num()
        self.lot_size = 1000
        self.saved_var = dict()  # saving the perturbed gradients
        for tensor_name, tensor in self.model.named_parameters():
            self.saved_var[tensor_name] = torch.zeros_like(tensor)  # a container of intermediate perturbed grads

    def build_log_path(self):
        """
        build log file direction and return the log filepath
        """
        experiment_path = os.path.abspath(self.args.experiment_name)
        logs_fp = "{}/{}".format(experiment_path, "pure_train")
        if not os.path.exists(logs_fp):
            os.makedirs(logs_fp)
        return experiment_path, logs_fp

    def new_log_file(self):
        """
        build attributes of log file for training and testing
        """
        train_stat_names = ['epoch', 'avg_train_loss', 'train_acc_mean', 'train_correct', 'train_total',
                            'avg_val_loss', 'val_acc_mean', 'val_correct', 'val_total']
        test_stat_names = ['avg_test_loss', 'test_acc_mean', 'test_correct', 'test_total']

        train_fp = os.path.join(self.logs_fp, self.train_sum_name)
        test_fp = os.path.join(self.logs_fp, self.test_sum_name)

        if not (os.path.exists(train_fp) and os.path.exists(test_fp)):
            save_statistics(self.logs_fp, train_stat_names, filename=self.train_sum_name, create=True)
            save_statistics(self.logs_fp, test_stat_names, filename=self.test_sum_name, create=True)

    def get_grad_layer_num(self):
        # return the number of layers that have grad
        i = 0
        for tensor_name, tensor in self.model.named_parameters():
            if 'pool' not in tensor_name:
                i += 1
        return i

    def train(self, epoch, train_loader):
        """
        Model Training
        """
        print('Epoch {}/{}'.format(epoch + 1, self.epochs))
        print('-' * 10)
        start_time = time.time()
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print('no dp grad_norm: ', self.grad_norm())
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(input=outputs, target=targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('TrainLoss: %.3f ' % (train_loss/(batch_idx+1)))
        self.scheduler.step()
        avg_loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total
        end_time = time.time()

        stat = [epoch, avg_loss, acc, correct, total]
        #save_statistics(self.logs_fp, [avg_loss, acc, correct, total], filename="train_summary.csv")
        print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (
            avg_loss, acc, correct, total, end_time - start_time))
        return stat

    def grad_norm(self):
        """
        calculate the norm of the gradients of all the parameters of a model.
        """
        norm_type = 2.0
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        if norm_type == 'inf':
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def train_dp(self, epoch, train_loader):
        """
        Model Training with differential privacy
        """
        print('Epoch {}/{}'.format(epoch + 1, self.epochs))
        print('-' * 10)
        start_time = time.time()
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        print('batch size: {}, lot size: {}, noise multiplier: {}'.format(self.batch_size, self.lot_size, self.noise_multiplier))

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = F.cross_entropy(input=outputs, target=targets)
            loss.backward()
            if batch_idx > 0 and batch_idx % self.lot_size == 0:
                print('grad_norm: ', self.grad_norm())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.S)  # per-layer clip on the grad
            for tensor_name, tensor in self.model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    self.saved_var[tensor_name].add_(new_grad)  # cal the sum of the clipped grad of diff task

            self.optimizer.zero_grad()

            noise_size = self.S * self.noise_multiplier #*np.sqrt(self.grad_layer_num) *
            if batch_idx > 0 and batch_idx % self.lot_size == 0:
                print(' sample_idx/samples: {}/{}'.format(batch_idx, len(train_loader)))
                for tensor_name, tensor in self.model.named_parameters():
                    if tensor.grad is not None:
                        if self.use_cuda:
                            self.saved_var[tensor_name].add_(
                                torch.FloatTensor(tensor.grad.shape).normal_(0, noise_size).to(self.device))
                        else:
                            self.saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, noise_size))
                        tensor.grad = self.saved_var[tensor_name] / (self.lot_size * self.batch_size)  # average
                print('grad_norm after perturbations: ', self.grad_norm())
                self.optimizer.step()

                # reset the saved_var as all zeros
                for tensor_name, tensor in self.model.named_parameters():
                    self.saved_var[tensor_name] = torch.zeros_like(tensor)

                avg_loss = train_loss / (batch_idx + 1)
                acc = 100. * correct / total
                print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        #self.scheduler.step()
        avg_loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total
        end_time = time.time()

        stat = [epoch, avg_loss, acc, correct, total]
        print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (
            avg_loss, acc, correct, total, end_time - start_time))
        return stat

    def test(self, model, val_loader):
        """
        Model Testing or evaluation
        param: model: if model is None evaluate the current model, else evaluate the input model
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if model is None:
            test_model = self.model
        else:
            test_model = model

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = test_model(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = test_loss / (batch_idx + 1)
        stat = [avg_loss, acc, correct, total]
        if model is None:
            print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))
            # Save checkpoint.
            if acc > self.best_acc:
                print('Saving..')
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(self.model.state_dict(), './checkpoint/ckpt.pth')
                self.best_acc = acc
        else:
            save_statistics(self.logs_fp, stat, filename=self.test_sum_name)
            print('\nThe Best Result>>>>>>>TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))
        return stat

    def load_model(self, model_file_path, pretrain_model=False, rm_fc_layer=False):
        """
        load parameters of pre_train model
        :param model_file_path: file path of the pre_train_model.
        :param remove_fc_layer: a flag indicating whether to remove the fully connected layer
        """
        state = torch.load(model_file_path, map_location='cuda:0')
        if pretrain_model:
            param_dict_new = state
        else:
            param_dict_new = dict()
            if 'network' in state.keys():
                param_dict = state['network']
            else:
                param_dict = state

            with open('./param_relationship.json', mode="r") as f:
                load_dict = json.load(fp=f)
            for meta_p, p in load_dict.items():
                if 'norm_layer' in meta_p and len(list(param_dict[meta_p].shape)) > 1:
                    param_dict_new[p] = param_dict[meta_p][0]
                else:
                    param_dict_new[p] = param_dict[meta_p]

        if rm_fc_layer:
            del param_dict_new['classifier.weight']
            del param_dict_new['classifier.bias']

        self.model.load_state_dict(state_dict=param_dict_new, strict=False)

    def run_pretrain(self, dp=False):
        """
        organize the experiment run, validation and test
        """
        self.new_log_file()
        if dp:
            self.batch_size = 1

        test_loader, _ = data_loader(self.test_data, batch_size=self.batch_size, shuffle=True)
        train_loader, val_loader = data_loader(self.train_data, batch_size=self.batch_size,
                                               shuffle=True, split_val=False)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if dp:
                train_stat = self.train_dp(epoch, train_loader=train_loader)
            else:
                train_stat = self.train(epoch, train_loader=train_loader)
            # val_stat = self.test(model=None, val_loader=val_loader)

            test_stat = self.test(model=None, val_loader=test_loader)
            train_stat.extend(test_stat)
            save_statistics(self.logs_fp, train_stat, filename=self.train_sum_name)

        # Loading weight files to the model and testing them.
        net_test = self.model
        net_test = net_test.to(self.device)
        net_test.load_state_dict(torch.load('./checkpoint/ckpt.pth'))
        torch.save(net_test.state_dict(), './checkpoint/pretrain.pth')
        acc = self.test(model=net_test, val_loader=test_loader)
        return acc

    def run_experiment(self, load_pretrain_fp=None, pretrain_model=False, rm_fc_layer=False):
        """
        organize the experiment run, validation and test
        """
        if load_pretrain_fp is not None:
            self.load_model(load_pretrain_fp, pretrain_model=pretrain_model, rm_fc_layer=rm_fc_layer)
            print(os.path.basename(load_pretrain_fp).split('.'))
            self.train_sum_name = str(os.path.basename(load_pretrain_fp).split('.')[0]) + 'train_summary.csv'
            self.test_sum_name = str(os.path.basename(load_pretrain_fp).split('.')[0]) + 'test_summary.csv'
            print(self.test_sum_name)

        self.new_log_file()
        for layer, tensor in self.model.named_parameters():
            if 'norm' in layer:
                tensor.requires_grad = False

        test_loader, _ = data_loader(self.test_data, batch_size=self.batch_size, shuffle=True)

        if self.full_batch:
            self.batch_size = len(self.train_data)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            train_loader, _ = data_loader(self.train_data, batch_size=self.batch_size,
                                          shuffle=False, split_val=False)

            train_stat = self.train(epoch, train_loader=train_loader)
            val_stat = self.test(model=None, val_loader=test_loader)
            train_stat.extend(val_stat)
            save_statistics(self.logs_fp, train_stat, filename=self.train_sum_name)

        # Loading weight files to the model and testing them.
        net_test = self.model
        net_test = net_test.to(self.device)
        net_test.load_state_dict(torch.load('./checkpoint/ckpt.pth'))

        test_stat = self.test(model=net_test, val_loader=test_loader)
        acc = test_stat[1]
        return acc


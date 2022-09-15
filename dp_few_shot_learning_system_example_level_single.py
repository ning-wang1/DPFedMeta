import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from utils.dp_utils import clip_grad_norm_for_autograd, dict_norm
from utils.storage import build_experiment_folder, save_to_json, update_json_experiment_log_dict
from utils.dp_utils import list_norm
from utils.dp_utils import get_percentile


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifierExampleSingle(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifierExampleSingle, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)
        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name, dp=True)
        self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                             num_classes_per_set,
                                             args=args, device=device, meta_classifier=True).to(device=self.device)
        self.task_learning_rate = args.task_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))

        self.inner_loop_optimizer_1 = LSLRGradientDescentLearningRule(device=device,
                                                                      init_learning_rate=0.15,
                                                                      total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                      use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer_1.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.classifier.named_parameters()))

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        # parameters that is related to Differential Privacy
        self.S = self.args.clip_norm_init
        # if os.path.exists("./experiment_config/clip_norms.json"):
        #     print('----------------------clipping Threshold Decision base on history')
        #     self.S = get_percentile("./experiment_config/clip_norms.json", 50)/np.sqrt(self.grad_layer_num)
        self.inner_S = 0.4                                  # the initialized inner sensitivity
        self.noise_multiplier = self.args.noise_multiplier  # the noise multiplier. real noise level=sigma*S
        self.lot_size = self.args.lot_size                  # lot_size*batch_size=number of tasks TO add noise

        # adaptive clip is False for example level
        self.adaptive_clip = args.adaptive_clip             # a flag indicating adaptive clip or constant clip
        self.clip_percentile = self.args.clip_percentile
        self.norm_list = []                                 # grad_norm, used to cal the sensitivity

        self.saved_var = dict()                             # saving the perturbed gradients
        for tensor_name, tensor in self.classifier.named_parameters():
            self.saved_var[tensor_name] = torch.zeros_like(tensor)  # a container of intermediate perturbed grads
        self.dp = True
        self.grad_layer_num = 0

        self.loss_ls = []
        self.grads_sum = dict()

    def get_grad_layer_num(self):
        # return the number of layers that have grad
        i = 0
        for tensor_name, tensor in self.classifier.named_parameters():
            if 'pool' not in tensor_name:
                i += 1
        return i

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def get_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters.
        :param params:  network's parameters.
        :return: A dictionary of the parameters.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                param_dict[name] = param.to(device=self.device)
        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        losses = loss                               # a list of losses for several records
        grads_sum = dict()

        for i, loss in enumerate(losses):           # an item is the loss of a record
            self.classifier.zero_grad(names_weights_copy)
            grads = torch.autograd.grad(loss, names_weights_copy.values(), retain_graph=True,
                                        create_graph=use_second_order)
            
            for pos, name in enumerate(names_weights_copy.keys()):
                clipped_grads = clip_grad_norm_for_autograd(grads[pos], self.inner_S)
                if name in grads_sum.keys():
                    grads_sum[name].add_(clipped_grads[0])
                else:
                    grads_sum[name] = clipped_grads[0]

        noise_size = self.inner_S * np.sqrt(len(names_weights_copy.keys())) * self.args.inner_z
        for name in names_weights_copy.keys():
            if self.use_cuda:
                grads_sum[name].add_(
                    torch.cuda.FloatTensor(grads_sum[name].shape).normal_(0, noise_size).to(self.device))
            else:
                grads_sum[name].add_(
                    torch.FloatTensor(grads_sum[name].shape).normal_(0, noise_size))
            grads_sum[name] = grads_sum[name] / len(losses)     # average the grad sum

        names_weights_copy = self.inner_loop_optimizer_1.update_params(names_weights_dict=names_weights_copy,
                                                                       names_grads_wrt_params_dict=grads_sum,
                                                                       num_step=current_step_idx)
        return names_weights_copy

    def apply_inner_loop_update_1(self, loss, names_weights_copy, use_second_order, current_step_idx, sample_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """

        self.classifier.zero_grad(names_weights_copy)
        grads = torch.autograd.grad(loss, names_weights_copy.values(), retain_graph=True,
                                    create_graph=use_second_order)

        for pos, name in enumerate(names_weights_copy.keys()):
            clipped_grads = clip_grad_norm_for_autograd(grads[pos], self.inner_S)
            if name in self.grads_sum.keys():
                self.grads_sum[name].add_(clipped_grads[0])
            else:
                self.grads_sum[name] = clipped_grads[0]

        if sample_idx == self.args.num_samples_per_class * self.args.num_classes_per_set - 1:
            noise_size = self.inner_S * np.sqrt(len(names_weights_copy.keys())) * self.args.inner_z
            for name in names_weights_copy.keys():
                if self.use_cuda:
                    self.grads_sum[name].add_(
                        torch.cuda.FloatTensor(self.grads_sum[name].shape).normal_(0, noise_size).to(self.device))
                else:
                    self.grads_sum[name].add_(
                        torch.FloatTensor(self.grads_sum[name].shape).normal_(0, noise_size))
                self.grads_sum[name] = self.grads_sum[name] / self.args.num_samples_per_class * self.args.num_classes_per_set  # average the grad sum

            names_weights_copy = self.inner_loop_optimizer_1.update_params(names_weights_dict=names_weights_copy,
                                                                           names_grads_wrt_params_dict=self.grads_sum,
                                                                           num_step=current_step_idx)
            self.grads_sum.clear()
        return names_weights_copy

    def apply_inner_loop_update_no_dp(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        loss = torch.mean(loss)
        self.classifier.zero_grad(names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order)
        names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_wrt_params,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            for num_step in range(num_steps):
                if training_phase:
                    for i in range(x_support_set_task.shape[0]):
                        x = x_support_set_task[i]
                        y = y_support_set_task[i]
                        x = x.view(-1, c, h, w)
                        y = y.view(-1)
                        if i == self.args.num_samples_per_class * self.args.num_classes_per_set - 1:
                            use_second = True
                        else:
                            use_second = False
                        support_loss, support_preds = self.net_forward(x=x,
                                                                       y=y,
                                                                       weights=names_weights_copy,
                                                                       backup_running_statistics=
                                                                       True if (num_step == 0) else False,
                                                                       training=True, num_step=num_step)
                        names_weights_copy = self.apply_inner_loop_update_1(loss=support_loss,
                                                                            names_weights_copy=names_weights_copy,
                                                                            use_second_order=use_second,
                                                                            current_step_idx=num_step,
                                                                            sample_idx=i)
                else:
                    support_loss, support_preds = self.net_forward(x=x_support_set_task,
                                                                   y=y_support_set_task,
                                                                   weights=names_weights_copy,
                                                                   backup_running_statistics=
                                                                   True if (num_step == 0) else False,
                                                                   training=True, num_step=num_step)
                    names_weights_copy = self.apply_inner_loop_update_no_dp(loss=support_loss,
                                                                            names_weights_copy=names_weights_copy,
                                                                            use_second_order=use_second_order,
                                                                            current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                     y=y_target_set_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_step)
                        task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses), dim=0)        # sum on steps
            total_losses.append(task_losses)                                # a list of losses, one element for one task
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = dict()
        losses['loss'] = torch.reshape(torch.stack(total_losses), (-1, 1))  # a list of losses of examples
        losses['accuracy'] = np.mean(total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)
        loss = F.cross_entropy(input=preds, target=y, reduce=False)  # reduce=false output is per_example loss
        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def run_train_iter(self, data_batch, epoch, iter_n):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        # # no dp on the outer loop
        # self.train_none_dp(losses['loss'], iter_n)

        # adding differential privacy to the outer-loop (meta model)
        self.train_dp(losses['loss'], iter_n)

        if iter_n > 0 and iter_n % self.lot_size == 0:
            losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        losses['loss'] = torch.mean(losses['loss'])  # cal the sum of loss of different tasks

        return losses, per_task_target_preds

    def train_none_dp(self, loss, iter_n):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        :para iter_n: The current iteration number
        """
        self.loss_ls.append(loss)
        if iter_n > 0 and iter_n % self.lot_size == 0:
            self.optimizer.zero_grad()
            loss = torch.mean(torch.stack(self.loss_ls))
            loss.backward()
            if 'imagenet' in self.args.dataset_name:
                for name, param in self.classifier.named_parameters():
                    if param.requires_grad:
                        param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
            print('gradient norm-----------------{}'.format(self.grad_norm()))
            self.optimizer.step()
            self.loss_ls = []

    def train_dp(self, losses, iter_n):
        """
        Applies an outer loop update with dp on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        :para iter_n: The current iteration number
        """
        for pos, j in enumerate(losses):                            # an item is the loss of a record
            self.optimizer.zero_grad()
            j.backward(retain_graph=True)
            for tensor_name, tensor in self.classifier.named_parameters():
                if tensor.grad is not None:
                    torch.nn.utils.clip_grad_norm_(tensor, self.S)
                    new_grad = tensor.grad
                    self.saved_var[tensor_name].add_(new_grad)      # cal the sum of the clipped grad
            
            self.optimizer.zero_grad()

        self.grad_layer_num = self.get_grad_layer_num()
        noise_size = self.S * np.sqrt(self.grad_layer_num) * self.noise_multiplier
        num_per_lot = self.lot_size * self.batch_size * self.args.num_target_samples * self.args.num_samples_per_class

        if iter_n > 0 and iter_n % self.lot_size == 0:
            for tensor_name, tensor in self.classifier.named_parameters():
                if tensor.grad is not None:
                    if self.use_cuda:
                        self.saved_var[tensor_name].add_(
                            torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, noise_size).to(self.device))
                    else:
                        self.saved_var[tensor_name].add_(
                            torch.FloatTensor(tensor.grad.shape).normal_(0, noise_size))  # add noise
                    # tensor.grad = self.saved_var[tensor_name] / (
                    #         self.lot_size * self.batch_size)  # average the grad sum
                    tensor.grad = self.saved_var[tensor_name] / num_per_lot
            print('\n gradient norm--------{}'.format(self.grad_norm()))
            self.optimizer.step()

            # reset the saved_var as all zeros
            for tensor_name, tensor in self.classifier.named_parameters():
                self.saved_var[tensor_name] = torch.zeros_like(tensor)

    def grad_norm(self):
        # calculate the norm of the gradients of all the parameters of a model.
        norm_type = 2.0
        parameters = self.classifier.parameters()
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

    def save_clip_norms(self, clip_s):
        """
        keep a record of the adaptive cliping process
        clip_s: the current clipping norm
        """
        clip_norms = dict()
        filename = os.path.join(self.logs_filepath, "clip_norms.json")

        if not os.path.exists(filename):
            clip_norms['clip_s'] = clip_s
            save_to_json(filename, clip_norms)
        else:
            update_json_experiment_log_dict('clip_s', clip_s, self.logs_filepath, log_name="clip_norms.json")

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        losses['loss'] = torch.mean(losses['loss'])  # cal the sum of loss of different tasks

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state

#
# for i in range(x_support_set_task.shape[0]):
#     x = x_support_set_task[i]
#     y = y_support_set_task[i]
#     x = x.view(-1, c, h, w)
#     y = y.view(-1)
#     support_loss, support_preds = self.net_forward(x=x,
#                                                    y=y,
#                                                    weights=names_weights_copy,
#                                                    backup_running_statistics=
#                                                    True if (num_step == 0) else False,
#                                                    training=True, num_step=num_step)
#     if training_phase:
#         names_weights_copy = self.apply_inner_loop_update_1(loss=support_loss,
#                                                             names_weights_copy=names_weights_copy,
#                                                             use_second_order=use_second_order,
#                                                             current_step_idx=num_step,
#                                                             sample_idx=i)
#         # names_weights_copy = self.apply_inner_loop_update_no_dp(loss=support_loss,
#         #                                                         names_weights_copy=names_weights_copy,
#         #                                                         use_second_order=use_second_order,
#         #                                                         current_step_idx=num_step)
#         # names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
#         #                                                   names_weights_copy=names_weights_copy,
#         #                                                   use_second_order=use_second_order,
#         #                                                   current_step_idx=num_step)
#     else:
#         names_weights_copy = self.apply_inner_loop_update_no_dp(loss=support_loss,
#                                                                 names_weights_copy=names_weights_copy,
#                                                                 use_second_order=use_second_order,
#                                                                 current_step_idx=num_step)
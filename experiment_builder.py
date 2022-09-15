import tqdm
import os
import numpy as np
import math
import sys
from utils.storage import build_experiment_folder, save_statistics, save_to_json, update_json_experiment_log_dict
import time
from tfcode.rdp_accountant import compute_rdp, get_privacy_spent
from compute_dp_sgd_privacy import apply_dp_sgd_analysis


class ExperimentBuilder(object):
    def __init__(self, args, data, model, device, example_level=False):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """
        self.args, self.device = args, device
        self.example_level = example_level

        self.model = model
        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name, dp=self.model.dp)
        self.model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "model_random"), state={})
        print("saved models to", self.saved_models_filepath)

        self.total_losses = dict()
        self.state = dict()
        self.state['best_val_acc'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.state['current_iter'] = 0
        self.start_epoch = 0
        self.max_models_to_save = self.args.max_models_to_save
        self.create_summary_csv = False
        self.iteration_constrained = self.args.total_epochs * self.args.total_iter_per_epoch

        if self.args.continue_from_epoch == 'from_scratch':
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == 'latest':
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            print("attempting to find existing checkpoint", )
            if os.path.exists(checkpoint):
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx='latest')
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

            else:
                self.args.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True
        elif int(self.args.continue_from_epoch) >= 0:
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=self.args.continue_from_epoch)
            self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

        self.data = data(args=args, current_iter=self.state['current_iter'])

        print("train_seed {}, val_seed: {}, at start time".format(self.data.dataset.seed["train"],
                                                                  self.data.dataset.seed["val"]))
        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
        self.epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)
        self.augment_flag = True if 'omniglot' in self.args.dataset_name.lower() else False
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        print(self.state['current_iter'], int(self.args.total_iter_per_epoch * self.args.total_epochs))

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses:
            summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])
            summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
            if key in ["loss", "accuracy"]:
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train):
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider
        :param sample_idx: The index of the incoming sample, in relation to the current training run.
        :param epoch_idx: The epoch index.
        :param total_losses: The current total losses dictionary to be updated.
        :param current_iter: The current training iteration in relation to the whole experiment.
        :param pbar_train: The progress bar of the training.
        :return: Updates total_losses, train_losses, current_iter
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = train_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        if sample_idx == 0:
            print("shape of data", x_support_set.shape, x_target_set.shape, y_support_set.shape,
                  y_target_set.shape)

        losses, _ = self.model.run_train_iter(data_batch=data_batch, epoch=epoch_idx, iter_n=current_iter)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)
        pbar_train.update(1)
        pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))      # pbar update

        current_iter += 1

        return train_losses, total_losses, current_iter

    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        losses, _ = self.model.run_validation_iter(data_batch=data_batch)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        pbar_val.set_description(
            "val_phase {} -> {}".format(self.epoch, val_output_update))  # print per-batch

        return val_losses, total_losses

    def test_evaluation_iteration(self, val_sample, model_idx, sample_idx, per_model_per_batch_preds, pbar_test):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_test: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_preds = self.model.run_validation_iter(data_batch=data_batch)

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        test_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        pbar_test.set_description(
            "test_phase {} -> {}".format(self.epoch, test_output_update))

        return per_model_per_batch_preds

    def save_models(self, model, epoch, state):
        """
        Saves two separate instances of the current model. One to be kept for history and reloading later and another
        one marked as "latest" to be used by the system for the next epoch training. Useful when the training/val
        process is interrupted or stopped. Leads to fault tolerant training and validation systems that can continue
        from where they left off before.
        :param model: Current meta learning model of any instance within the few_shot_learning_system.py
        :param epoch: Current epoch
        :param state: Current model and experiment state dict.
        """
        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)
        # for name, tensor in state['network'].items():
        #     print('nnnnnn', name)
        #     print(tensor.shape)

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

        print("saved models to", self.saved_models_filepath)

    def pack_and_save_metrics(self, start_time, create_summary_csv, train_losses, val_losses, state):
        """
        Given current epochs start_time, train losses, val losses and whether to create a new stats csv file, pack stats
        and save into a statistics csv file. Return a new start time for the new epoch.
        :param start_time: The start time of the current epoch
        :param create_summary_csv: A boolean variable indicating whether to create a new statistics file or
        append results to existing one
        :param train_losses: A dictionary with the current train losses
        :param val_losses: A dictionary with the currrent val loss
        :return: The current time, to be used for the next epoch.
        """
        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)

        if 'per_epoch_statistics' not in state:
            state['per_epoch_statistics'] = dict()

        for key, value in epoch_summary_losses.items():

            if key not in state['per_epoch_statistics']:
                state['per_epoch_statistics'][key] = [value]
            else:
                state['per_epoch_statistics'][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time, state

    def evaluated_test_set_using_the_best_models(self, top_n_models):
        per_epoch_statistics = self.state['per_epoch_statistics']
        val_acc = np.copy(per_epoch_statistics['val_accuracy_mean'])
        val_idx = np.array([i for i in range(len(val_acc))])
        sorted_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1][:top_n_models]

        sorted_val_acc = val_acc[sorted_idx]
        val_idx = val_idx[sorted_idx]
        print(sorted_idx)
        print(sorted_val_acc)

        top_n_idx = val_idx[:top_n_models]
        per_model_per_batch_preds = [[] for i in range(top_n_models)]
        per_model_per_batch_targets = [[] for i in range(top_n_models)]
        test_losses = [dict() for i in range(top_n_models)]
        for idx, model_idx in enumerate(top_n_idx):
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=model_idx + 1)
            with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                   augment_images=False)):
                    # print(test_sample[4])
                    per_model_per_batch_targets[idx].extend(np.array(test_sample[3]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                               sample_idx=sample_idx,
                                                                               model_idx=idx,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)
        # for i in range(top_n_models):
        #     print("test assertion", 0)
        #     print(per_model_per_batch_targets[0], per_model_per_batch_targets[i])
        #     assert np.equal(np.array(per_model_per_batch_targets[0]), np.array(per_model_per_batch_targets[i]))

        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)  # average among models
        # print(per_batch_preds.shape)
        per_batch_max = np.argmax(per_batch_preds, axis=2)   # 
        per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(per_batch_max.shape)  
        #print(per_batch_max)
        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}

        _ = save_statistics(self.logs_filepath,
                            list(test_losses.keys()),
                            create=True, filename="test_summary.csv")

        summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                      list(test_losses.values()),
                                                      create=False, filename="test_summary.csv")
        print(test_losses)
        print("saved test performance at", summary_statistics_filepath)

    def run_experiment_test_only(self):
        self.evaluated_test_set_using_the_best_models(top_n_models=5)

    def random_initialize_test(self):
        print('\n RANDOM INITIALIZATION EVALUATION---------------\n')
        per_model_per_batch_preds = [[]]
        per_model_per_batch_targets = [[]]

        self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="model", model_idx='random')
        with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
            for sample_idx, test_sample in enumerate(
                    self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                               augment_images=False)):

                per_model_per_batch_targets[0].extend(np.array(test_sample[3]))
                per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                           sample_idx=sample_idx,
                                                                           model_idx=0,
                                                                           per_model_per_batch_preds=per_model_per_batch_preds,
                                                                           pbar_test=pbar_test)
        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)  # average among models
        per_batch_max = np.argmax(per_batch_preds, axis=2)  #
        per_batch_targets = np.array(per_model_per_batch_targets).reshape(per_batch_max.shape)
        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}
        print(test_losses)

    def evaluate_test_set(self):
        per_epoch_statistics = self.state['per_epoch_statistics']
        val_acc = np.copy(per_epoch_statistics['val_accuracy_mean'])
        val_idx = np.array([i for i in range(len(val_acc))])

        _ = save_statistics(self.logs_filepath,
                            ["test_accuracy_mean", "test_accuracy_std"],
                            create=True, filename="test_summary_per_epoch.csv")

        for idx, model_idx in enumerate(val_idx):
            per_model_per_batch_preds = [[]]
            per_model_per_batch_targets = [[]]

            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=model_idx + 1)
            with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                   augment_images=False)):

                    per_model_per_batch_targets[0].extend(np.array(test_sample[3]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                               sample_idx=sample_idx,
                                                                               model_idx=0,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)

            per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)  # average among models
            per_batch_max = np.argmax(per_batch_preds, axis=2)   #
            per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(per_batch_max.shape)
            accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
            accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

            test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}
            summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                          list(test_losses.values()),
                                                          create=False, filename="test_summary_per_epoch.csv")
            print(test_losses)
            print("saved test performance at", summary_statistics_filepath)

    def track_dp_task_level(self, current_iter):
        dp = dict()
        try:
            total_iters = self.iteration_constrained
        except:
            total_iters = self.args.total_iter_per_epoch * self.args.total_epochs

        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])

        steps = int(current_iter / self.model.lot_size)
        q = self.model.lot_size/total_iters  # sampling probability
        z = self.model.noise_multiplier

        if steps > 0:
            dp_info = apply_dp_sgd_analysis(q, z, steps, orders, 1e-6)  # calculate the dp parameters
            filename = os.path.join(self.logs_filepath, "dp_track.json")
            if not os.path.exists(filename):
                dp['step'] = steps
                dp['eps'] = dp_info['eps']
                save_to_json(filename, dp)
            else:
                update_json_experiment_log_dict('eps', dp_info['eps'], self.logs_filepath, log_name="dp_track.json")
                update_json_experiment_log_dict('step', steps, self.logs_filepath, log_name="dp_track.json")

    def run_rdp_compute(self, eps_target):
        N = self.args.total_iter_per_epoch * self.args.batch_size * self.args.total_epochs  # number of tasks
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])
        tasks_per_step = self.model.lot_size * self.args.batch_size
        q = tasks_per_step / N  # q - the sampling ratio.
        z = self.model.noise_multiplier
        steps_low_bound = 0
        steps_high_bound = int(N/tasks_per_step)
        steps = 1

        # The following code is a binary search algorithm, find the steps that does not violate the target epsilon
        while steps_high_bound - steps_low_bound > 1:
            steps = int((steps_low_bound + steps_high_bound) / 2)
            dp_info = apply_dp_sgd_analysis(q, z, steps, orders, 1e-6)
            eps = dp_info['eps']
            if eps > eps_target:
                steps_high_bound = steps
            else:
                steps_low_bound = steps

        N_train = steps * tasks_per_step
        if N_train > N:
            N_train = N
        self.iteration_constrained = N_train / self.args.batch_size

        hyper_impact_dp = dict()
        hyper_impact_dp['lot size'] = self.model.lot_size
        hyper_impact_dp['meta learning rate'] = self.model.args.meta_learning_rate
        hyper_impact_dp['epoch'] = N / (self.args.total_iter_per_epoch * self.args.batch_size)
        hyper_impact_dp['tasks'] = N
        hyper_impact_dp['clip_percentile'] = self.model.clip_percentile
        save_to_json(filename=os.path.join(self.logs_filepath, "privacy parameters.json"),
                     dict_to_store=[dp_info, hyper_impact_dp])

    def run_rdp_compute_example_level(self, eps_target, dataset):
        if 'cifar' or 'imageNet' in dataset:
            N = 600 * 100
        else:
            N = 1623*20  # number of examples

        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                  list(range(5, 64)) + [128, 256, 512])
        z = self.model.noise_multiplier
        inner_z = self.args.inner_z
        delta = 1e-5

        # inner privacy loss
        examples_per_inner_step = self.args.num_samples_per_class * self.args.num_classes_per_set
        q_inner = examples_per_inner_step / N  # q - the sampling ratio.
        inner_steps_per_lot = self.args.batch_size * self.args.lot_size
        rdp_inner = compute_rdp(q_inner, inner_z, inner_steps_per_lot, orders)
        # outer privacy loss
        examples_per_outer_step = self.args.num_target_samples * self.args.num_classes_per_set * inner_steps_per_lot
        q_outer = examples_per_outer_step / N
        print('num_samples_per_class: {}, target: {}'.format(self.args.num_samples_per_class, self.args.num_target_samples))
        print('inner_steps_per_lot: {}, batch size: {}, lot_size: {}, steps: {}'.format(inner_steps_per_lot,
                                                                                        self.args.batch_size,
                                                                                        self.args.lot_size,
                                                                                        self.args.number_of_training_steps_per_iter))
        print('q_outer:, {},  z_outer: {}, q_inner: {}, z_inner: {}'.format(q_outer, z, q_inner, inner_z))
        rdp_outer = compute_rdp(q_outer, z, 1, orders)

        total_rdp_per_lot = rdp_inner + rdp_outer
        outer_steps_low_bound = 1
        outer_steps_high_bound = self.args.total_epochs * self.args.total_iter_per_epoch/self.args.lot_size *2
        # The following code is a binary search algorithm, find the steps that does not violate the target epsilon
        while outer_steps_high_bound - outer_steps_low_bound > 2:
            steps = int((outer_steps_low_bound + outer_steps_high_bound) / 2)
            rdp = total_rdp_per_lot * steps
            eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
            print('DP-SGD with outer sampling rate = {:.3g}%  inner sampling rate = {:.3g} % and noise_multiplier = {} iterated'
                  ' over {} steps satisfies'.format(100 * q_outer, 100* q_inner, z, steps), end=' ')
            print('differential privacy with eps = {:.3g} and delta = {}.'.format(
                eps, delta))
            print('The optimal RDP order is {}.'.format(opt_order))
            if eps > eps_target:
                outer_steps_high_bound = steps
            else:
                outer_steps_low_bound = steps

        N_train = outer_steps_low_bound
        self.iteration_constrained = N_train * self.args.lot_size

        hyper_impact_dp = dict()
        hyper_impact_dp['lot size'] = self.model.lot_size
        hyper_impact_dp['meta learning rate'] = self.model.args.meta_learning_rate
        hyper_impact_dp['epoch'] = N / (self.args.total_iter_per_epoch * self.args.batch_size)
        hyper_impact_dp['tasks'] = N
        hyper_impact_dp['clip_percentile'] = self.model.clip_percentile

        dp_info = dict()
        dp_info['sampling rate inner'] = q_inner
        dp_info['sampling rate outer'] = q_outer
        dp_info['noise_multiplier'] = z
        dp_info['step number'] = outer_steps_low_bound
        dp_info['eps'] = eps
        dp_info['delta'] = delta
        dp_info['optimal rdp order'] = opt_order
        save_to_json(filename=os.path.join(self.logs_filepath, "privacy parameters.json"),
                     dict_to_store=[dp_info, hyper_impact_dp])

    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
        will return the test set evaluation results on the best performing validation model.
        """
        try:
            with tqdm.tqdm(initial=self.state['current_iter'], ascii=True,
                           total=int(self.iteration_constrained)) as pbar_train:

                while (self.state['current_iter'] < self.iteration_constrained) and (
                        self.args.evaluate_on_test_set_only == False):

                    for train_sample_idx, train_sample in enumerate(
                            self.data.get_train_batches(total_batches=int(self.iteration_constrained) - self.state[
                                'current_iter'],
                                                        augment_images=self.augment_flag)):
                        # print(self.state['current_iter'], (self.args.total_epochs * self.args.total_iter_per_epoch))
                        train_losses, total_losses, self.state['current_iter'] = self.train_iteration(
                            train_sample=train_sample,
                            total_losses=self.total_losses,
                            epoch_idx=(self.state['current_iter'] /
                                       self.args.total_iter_per_epoch),
                            pbar_train=pbar_train,
                            current_iter=self.state['current_iter'],
                            sample_idx=self.state['current_iter'])

                        if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:

                            total_losses = dict()
                            val_losses = dict()
                            with tqdm.tqdm(
                                    total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_val:
                                for _, val_sample in enumerate(
                                        self.data.get_val_batches(
                                            total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                            augment_images=False)):
                                    val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                         total_losses=total_losses,
                                                                                         pbar_val=pbar_val, phase='val')

                                if val_losses["val_accuracy_mean"] > self.state['best_val_acc']:
                                    print("\nBest validation accuracy", val_losses["val_accuracy_mean"])
                                    self.state['best_val_acc'] = val_losses["val_accuracy_mean"]
                                    self.state['best_val_iter'] = self.state['current_iter']
                                    self.state['best_epoch'] = int(
                                        self.state['best_val_iter'] / self.args.total_iter_per_epoch)

                            self.epoch += 1
                            self.state = self.merge_two_dicts(first_dict=self.merge_two_dicts(first_dict=self.state,
                                                                                              second_dict=train_losses),
                                                              second_dict=val_losses)

                            self.save_models(model=self.model, epoch=self.epoch, state=self.state)

                            self.start_time, self.state = self.pack_and_save_metrics(start_time=self.start_time,
                                                                                     create_summary_csv=self.create_summary_csv,
                                                                                     train_losses=train_losses,
                                                                                     val_losses=val_losses,
                                                                                     state=self.state)

                            self.total_losses = dict()

                            self.epochs_done_in_this_run += 1

                            save_to_json(filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                                         dict_to_store=self.state['per_epoch_statistics'])

                            if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                                print(
                                    "train_seed {}, val_seed: {}, at pause time".format(self.data.dataset.seed["train"],
                                                                                        self.data.dataset.seed["val"]))
                                sys.exit()
                            if self.model.dp and self.example_level:
                                self.track_dp_task_level(self.state['current_iter'])
                if self.example_level:
                    pass
                else:
                    self.evaluated_test_set_using_the_best_models(top_n_models=5)

        except KeyboardInterrupt:
            pbar_train.close()
            raise
        pbar_train.close()
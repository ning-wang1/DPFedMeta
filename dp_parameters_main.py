import os
import math
from utils.storage import build_experiment_folder, save_to_json, update_json_experiment_log_dict
from compute_dp_sgd_privacy import apply_dp_sgd_analysis
from tfcode.rdp_accountant import compute_rdp, get_privacy_spent
# to compute the privacy loss of a experiment
# this code can be executed separately

def track_dp_task_level(total_iters, current_iter, lot_size, z, target_eps, target_delta):
    dp = dict()
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])

    steps = int(current_iter / lot_size)
    q = lot_size / total_iters  # sampling probability

    if steps > 0:
        rdp = compute_rdp(q, z, steps, orders)

        target_eps, delta, opt_order = get_privacy_spent(orders, rdp, target_eps=target_eps, target_delta=target_delta)

        print('step: {}, delta: {}, eps: {}'.format(steps, delta, target_eps))

        _, logs_filepath, _ = build_experiment_folder(experiment_name='omniglot_1_8_0.1_64_5_0', dp=True)
        filename = os.path.join(logs_filepath, "dp_track.json")
        if not os.path.exists(filename):
            dp['step'] = steps
            dp['delta'] = delta
            save_to_json(filename, dp)
        else:
            update_json_experiment_log_dict('delta', delta, logs_filepath, log_name="dp_track.json")
            update_json_experiment_log_dict('step', steps, logs_filepath, log_name="dp_track.json")

def track_dp_epochs(target_eps,target_delta):
    total_epochs = 100
    total_iter_per_epoch = 500
    batch_size = 8
    lot_size = 200
    z =3

    total_iters = total_epochs * total_iter_per_epoch
    total_steps = int(total_iters/lot_size)

    for i in range(int(total_epochs/2)):
        i = (i+1) * 2
        current_iter = i * total_iter_per_epoch
        track_dp_task_level(total_iters, current_iter, lot_size, z,target_eps,target_delta)

def track_dp_conventional(target_eps, target_delta):
    total_epochs = 100
    L = 2000
    examples = 48000
    q = L/examples
    z = 6
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])

    steps = examples/L*total_epochs

    rdp = compute_rdp(q, z, steps, orders)

    target_eps, delta, opt_order = get_privacy_spent(orders, rdp, target_eps=target_eps, target_delta=target_delta)

    print('step: {}, delta: {}, eps: {}'.format(steps, delta, target_eps))


def dp_tasks_num(target_eps, target_delta):
    total_iter_per_epoch = 500
    lot_size = 200
    z = 1

    for i in range(48, 70):
        total_iters = i * 8000 / 8 # number of training tasks
        current_iter = total_iters
        print('iters: ', current_iter)
        track_dp_task_level(total_iters, current_iter, lot_size, z, target_eps, target_delta)


# track_dp_epochs(target_eps=0.22, target_delta=None)
# dp_tasks_num(target_eps)
track_dp_conventional(target_eps=None, target_delta=1e-5)








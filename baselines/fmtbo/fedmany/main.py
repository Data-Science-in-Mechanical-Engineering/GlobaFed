import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.servergp import ServerGP
from deploy.clientgp import ClientGP
from objectives import TASK_NAMES, create_objectives
from utils.constants import Benchmarks, Acquisitions
from utils.connection import online
from results import objective_optimum, save_regret_csv


def main(iter, args):
    print('=============== Run {} ==============='.format(iter))
    np.random.seed(args.seed + iter)

    # or number of evaluations (real fitness calculations)
    total_clients = args.total_clients
    clients_per_round = int(total_clients * args.C)
    # num_init_points = args.dim if args.dim % 2 == 0 else args.dim + 1
    num_init_points = args.initial_points
    total_rounds = args.Total_FE

    params = {'num_init_points': num_init_points,
              'iid': args.iid,
              'dim': args.dim,
              'knowledge_transfer': args.knowledge_transfer,
              'kt_agg_prob': args.kt_agg_prob,
              'gamma': args.gamma,
              'opt_obj': args.opt_obj,
              'regr': args.regr,
              'acq_type': args.acq_type,
              'next_method': args.next_method}

    # ======= Create Server ========
    pre_start = time.time()
    server = ServerGP(server_id=0,
                      params=params,
                      clients_per_round=clients_per_round,
                      enable_parallel=args.enable_parallel)

    # ======= Create Clients ========
    tasks = create_objectives(
        args.task,
        args.dim,
        total_clients,
        np.random.default_rng(args.seed + iter),
    )
    partitioned_bounds = [None] * total_clients
    all_clients = [ClientGP(client_id=i, group_id=0, params=params, obj_func=t, pb=partitioned_bounds[i])
                   for i, t in enumerate(tasks)]

    if args.knowledge_transfer:
        server.create_x_s(all_clients)
        server.share_x_s_to(all_clients)
    pre_end = time.time()
    time_per_round = [pre_end - pre_start] * num_init_points

    if args.warm_start:
        ws_client = all_clients.pop()
    # ======= Federated Bayesian Optimization ========
    start = time.time()
    for i in range(total_rounds):

        # check if the communication round is correct or not
        assert i == server.round
        start_time = time.time()
        print('--------- Round %d of %d: Training %d Clients ---------'
              % (i + 1, total_rounds, clients_per_round))

        server.select_clients(online(all_clients), num_clients=clients_per_round)
        server.train()
        server.update_model(agg_alg='fedavg')

        duration_time = time.time()-start_time
        time_per_round.append(duration_time)
        print('One communication round training time: %.4fs' % duration_time)

        if args.warm_start and i + 1 == 25:
            all_clients.append(ws_client)

    duration = time.time() - start
    print('Overall elapsed time : ', duration)

    # ======= save results ========
    if args.save_results:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        post_fix = f"_dim{args.dim}_{args.acq_type}"
        if args.knowledge_transfer:
            post_fix += "_kt"
        if args.warm_start:
            post_fix += "_warm_start"

        root_save_file = f'results_fbo_{args.task}' + post_fix
        save_dir = os.path.join(root_save_file, timestamp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        obj_data = {}
        obj_file_name = f"run{iter}_objective_best.csv"
        # Save combination data
        for c in all_clients:
            obj_data[f'agent_{c.client_id}_best_y'] = [
                c.obj_func.optimum
            ]
        if obj_data:
            obj_pd = pd.DataFrame(data=obj_data)
            obj_pd.to_csv(os.path.join(save_dir, obj_file_name))

        if args.warm_start:
            ws_client = all_clients.pop()

        file_name = 'run' + str(iter) + f"_dim{args.dim}"
        file_name += '_C' + str(clients_per_round) + '.' + str(total_clients)
        if args.knowledge_transfer:
            file_name += '_ktp' + str(args.kt_agg_prob)
        if args.iid:
            iid_file = 'iid'
        else:
            iid_file = 'niid' + str(args.np_per_dim)
        file_name += '_FE' + str(total_rounds) + '_' + args.acq_type + \
            str(args.gamma) + '_' + iid_file + '_fedavg.csv'
        record_file_name = os.path.join(save_dir, file_name)

        record_data = {}
        for c in all_clients:
            record_data.update({'agent_' + str(c.client_id): c.fitness_values})
        record_data.update({'time': time_per_round})
        record_pd = pd.DataFrame(data=record_data)
        record_pd.to_csv(record_file_name)

        if args.warm_start:
            ws_data = {'agent_' + str(ws_client.client_id): ws_client.fitness_values}
            ws_pd = pd.DataFrame(data=ws_data)
            ws_pd.to_csv(os.path.join(save_dir, "warm_start.csv"))

    expected_observations = args.initial_points + args.Total_FE
    result_clients = [
        client
        for client in all_clients
        if len(client.fitness_values) == expected_observations
    ]
    histories = [
        np.asarray(client.fitness_values, dtype=float).copy()
        for client in result_clients
    ]
    optima = [objective_optimum(client.obj_func) for client in result_clients]

    del server
    del all_clients
    return histories, optima


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--num-runs', '--num_runs', dest='num_runs', default=1,
                        type=int, help='number of runs')
    parser.add_argument('--enable_parallel', default=True,
                        action='store_true', help='if using parallel computing')
    parser.add_argument('--iid', default=True, action='store_true',
                        help='client dataset partition methods')
    parser.add_argument('--np_per_dim', default=2, type=int,
                        help='number of partitions per feature dimension')
    parser.add_argument('--rounds', '--Total_FE', dest='Total_FE', default=50, type=int,
                        help='total number of federated evaluation')
    parser.add_argument('--num-clients', '--total_clients', dest='total_clients',
                        default=10, type=int,
                        help='total number of clients')
    parser.add_argument('--C', default=1., type=float, help='connection ratio')
    parser.add_argument('--dim', default=10, type=int,
                        help='the number of dimensions of decision variables')
    parser.add_argument('--task', default="ackley_shifted", type=str,
                        choices=TASK_NAMES,
                        help='the task to do')
    parser.add_argument('--initial-points', default=50, type=int,
                        help='initial objective evaluations per client')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed used for the first run')
    parser.add_argument('--output-dir', default=Path('logs'), type=Path,
                        help='folder for the aggregate regret CSV')
    parser.add_argument('--warm_start', default=False,
                        action='store_true', help='Warm start.')

    parser.add_argument('--func', default='', type=str,
                        choices=Benchmarks, help='The training dataset')
    parser.add_argument('--opt_obj', default='minimize', type=str, choices=['minimize', 'maximize'],
                        help='minimizing or maximizing the objective')
    parser.add_argument('--regr', default='regr_constant', type=str, choices=['regr_constant', 'regr_linear'],
                        help='regression type')
    parser.add_argument('--next_method', default='GA', type=str, choices=['GA', 'sampling'],
                        help='The sampling method for getting the next decision variable')
    parser.add_argument('--acq_type', default='EI_w', type=str,
                        choices=Acquisitions, help='Acquisition function')
    parser.add_argument('--knowledge_transfer', default=True, action='store_true',
                        help='if sharing the shared server datasets for transfer learning')
    parser.add_argument('--kt_agg_prob', default=0.8, type=float,
                        help='the proportion of clients for global model aggregation')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='the trade-off parameters of federated ensemble acquisition function')
    parser.add_argument('--save-results', '--save_results', dest='save_results',
                        action='store_true',
                        help='also save the original per-run client traces')
    args = parser.parse_args()

    completed_runs = []
    run_optima = []
    for iter in range(0, args.num_runs):
        histories, optima = main(iter, args)
        completed_runs.append(histories)
        run_optima.append(optima)
    save_regret_csv(
        completed_runs,
        run_optima,
        task=args.task,
        dim=args.dim,
        initial_points=args.initial_points,
        output_dir=args.output_dir,
    )

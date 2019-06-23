from train_pg_f18 import train_PG
from multiprocessing import Process
import os
import time

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, nargs='*')
    parser.add_argument('--learning_rate', '-lr', type=float, nargs='*')


    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists('data'):
        os.makedirs('data')
    logdir = args.env_name + '_' + args.exp_name + '_' + time.strftime('%Y-%m-%d_%H-%M-%S')
    logdir = os.path.join('data', logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for bsz in args.batch_size:
        for lr in args.learning_rate:
            param_str = 'b'+str(bsz)+'_lr'+str(lr)
            cur_logdir = os.path.join(logdir, str(args.seed), 'b'+str(bsz)+'_lr'+str(lr))
            def train_func():
                train_PG(exp_name=args.exp_name + '_' + param_str,
                         env_name=args.env_name,
                         n_iter=args.n_iter,
                         gamma=args.discount,
                         min_timesteps_per_batch=bsz,
                         max_path_length=max_path_length,
                         learning_rate=lr,
                         reward_to_go=args.reward_to_go,
                         animate=False,
                         logdir=cur_logdir,
                         normalize_advantages=not(args.dont_normalize_advantages),
                         nn_baseline=args.nn_baseline,
                         seed=args.seed,
                         n_layers=args.n_layers,
                         size=args.size)
            p = Process(target=train_func, args=tuple())
            p.start()

if __name__ == "__main__":
    main()

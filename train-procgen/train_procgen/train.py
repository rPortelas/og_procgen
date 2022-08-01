import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from train_procgen.env_make_utils import make_env, setup_loca
from baselines import logger
from mpi4py import MPI
import argparse

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)


def train_fn(env_name, num_envs, num_test_envs, nsteps, nminibatches, distribution_mode, num_levels, start_level,
             timesteps_per_proc, locacoinrun_draw_bars, is_test_worker=False, log_dir='/tmp/procgen', comm=None, args=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    #nsteps = 256
    #nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    if args.loca_training:
        print("starting loca training")
        assert (args.phase_1_len + args.phase_2_len + args.phase_3_len) == (timesteps_per_proc/1e6),\
            "sum of loca phases not equal to total timesteps planned: {} vs {}".format(args.phase_1_len + args.phase_2_len + args.phase_3_len, timesteps_per_proc/1e6)
        loca_params = {"phase_1_len": args.phase_1_len,
                       "phase_2_len": args.phase_2_len,
                       "phase_3_len": args.phase_3_len}
        # launching phase 1 directly
        locacoinrun_reward_phase = 1
    else:
        loca_params = None
        locacoinrun_reward_phase = 0  # i.e. regular rewards

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    env_kwargs={'num_levels': num_levels,
                'start_level': start_level,
                'distribution_mode': distribution_mode,
                'locacoinrun_draw_bars': locacoinrun_draw_bars,
                'locacoinrun_reward_phase': locacoinrun_reward_phase}

    venv = make_env(num_envs, env_name=env_name, **env_kwargs)

    if num_test_envs > 0:  # adding test vectorized environment
        test_env_kwargs = env_kwargs.copy()
        # fix test set across experiments
        test_env_kwargs['start_level'] = 42
        test_env_kwargs['num_levels'] = num_test_envs
        test_venv = make_env(num_test_envs, env_name=env_name, **test_env_kwargs)  # fixed levels for testing
    else:
        test_venv = None

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    print(nminibatches)
    print(nsteps)
    logger.info("training")
    ppo2.learn(
        env=venv,
        eval_env=test_venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=0,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        loca_params=loca_params,
    )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=9_000_000)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--nb_test_episodes', type=int, default=0)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--locacoinrun_draw_bars', action='store_true', default=True)
    parser.add_argument('--no_locacoinrun_draw_bars', dest='locacoinrun_draw_bars', action='store_false')
    parser.add_argument('--normalize_rewards', action='store_true', default=False)
    parser.add_argument('--result_dir', default='/gpfsscratch/rech/imi/uxo14qj/storage/ogprocgen_results',  # os.path.join(os.getcwd()
                        help="Directory Path to store results (default: %(default)s)")

    #Loca parameters
    parser.add_argument('--loca_training', action='store_true', default=True)
    parser.add_argument('--no_loca_training', dest='loca_training', action='store_false')
    parser.add_argument('--phase_1_len', type=float, default=5)
    parser.add_argument('--phase_2_len', type=float, default=5)
    parser.add_argument('--phase_3_len', type=float, default=15)


    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    log_dir = args.result_dir + '/' + args.exp_name + '/' + str(args.start_level)
    print('logging data in: {}'.format(log_dir))
    print(args)

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(args.env_name,
        args.num_envs,
        args.nb_test_episodes,
        args.nsteps,
        args.nminibatches,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        args.locacoinrun_draw_bars,
        is_test_worker=is_test_worker,
        comm=comm,
        log_dir=log_dir, args=args)

if __name__ == '__main__':
    main()

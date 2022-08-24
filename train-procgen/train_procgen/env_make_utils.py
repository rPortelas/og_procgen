from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)

def get_venv_params(env):
    p = env.unwrapped.env.options
    print(type(env))
    if isinstance(env, VecNormalize):
        use_rnorm=True
    else:
        use_rnorm=False

    return {'num_levels': p['num_levels'], 'start_level': p['start_level'],
            'distribution_mode': "easy" if p['distribution_mode'] == 0 else "hard",
            'locacoinrun_draw_bars': p['locacoinrun_draw_bars']}, p['env_name'], use_rnorm

def setup_loca(env, test_env, phase):
    print("updating env and eval_env for phase {}".format(phase))

    # extract env parameters before re-making it with new loca phase
    env_kwargs, env_name, use_rnorm = get_venv_params(env)
    env_kwargs['locacoinrun_reward_phase'] = phase
    env_kwargs['locacoinrun_restrict_spawn'] = True if phase == 2 else False
    if env_kwargs['locacoinrun_restrict_spawn']:
        print('restricting spawn for phase 2')
    # recreating env
    new_env = make_env(env.num_envs, env_name=env_name, normalize=use_rnorm, **env_kwargs)

    # extract test env parameters before re-making it with new loca phase
    test_env_kwargs, env_name, use_rnorm = get_venv_params(test_env)
    test_env_kwargs['locacoinrun_reward_phase'] = phase
    # do not restrict spawn in test envs

    # recreating test env
    new_test_env = make_env(test_env.num_envs, env_name=env_name, normalize=use_rnorm, **test_env_kwargs)

    env.close()
    test_env.close()
    return new_env, new_test_env


def make_env(num_envs, env_name, normalize=False, **kwargs):

    print(f"creating vectorized {env_name} environment with num_envs={num_envs} and \n {kwargs}")

    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, **kwargs)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    if normalize:
        print("normalizing rewards")
        venv = VecNormalize(venv=venv, ob=False)

    return venv
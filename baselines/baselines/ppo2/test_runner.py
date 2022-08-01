import numpy as np
from baselines.common.runners import AbstractEnvRunner

class TestRunner(AbstractEnvRunner):
    """
    We use this object to perform 1 test episode accross all vectorized test environments
    __init__:
    - Initialize the runner

    run():
    - Make a 1-episode test for all envs in the venv
    """
    def __init__(self, *, feed_shape, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

        # Sending this action to a procgen env does not change the env state
        self.idle_action = -2

        self.feed_shape = feed_shape  # this is the number of obs sent in parallel during training
        assert self.nenv >= self.feed_shape
        assert (self.nenv % self.feed_shape) == 0
        self.nsplits = (self.nenv // self.feed_shape)
        self.split_size = self.nenv // self.nsplits


    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        while not all(self.dones):
        #for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            # split inputs to fit training env dimensions
            if self.nsplits > 1:
                actions = None
                values = None
                neglogpacs = None
                dones_splits = [self.dones[i:i + self.split_size] for i in range(0, len(self.dones), self.split_size)]
                states_splits = [None]*self.nsplits if self.states is None else np.split(self.states, self.nsplits)
                for split_nb, (obs_split, states_split, dones_split) in enumerate(zip(np.split(self.obs, self.nsplits), states_splits, dones_splits)):
                    actions_split, _, _, _ = self.model.step(obs_split, S=states_split, M=dones_split)
                    if split_nb == 0:
                        actions = actions_split
                    else:
                        actions = np.concatenate((actions, actions_split))
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            # mb_obs.append(self.obs.copy())
            # mb_actions.append(actions)
            # mb_values.append(values)
            # mb_neglogpacs.append(neglogpacs)
            # mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            # Add idle action mask TODO
            actions[self.dones] = self.idle_action
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for idx, info in enumerate(infos):
                if actions[idx] != self.idle_action:  # ignore info from envs that are already finished
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        print(f"test_env{idx}:{maybeepinfo}")
                        epinfos.append(maybeepinfo)

            #mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        # mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        # mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        # mb_actions = np.asarray(mb_actions)
        # mb_values = np.asarray(mb_values, dtype=np.float32)
        # mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        # mb_dones = np.asarray(mb_dones, dtype=np.bool)
        # last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = None
        # mb_returns = np.zeros_like(mb_rewards)
        # mb_advs = np.zeros_like(mb_rewards)
        # lastgaelam = 0
        # for t in reversed(range(self.nsteps)):
        #     if t == self.nsteps - 1:
        #         nextnonterminal = 1.0 - self.dones
        #         nextvalues = last_values
        #     else:
        #         nextnonterminal = 1.0 - mb_dones[t+1]
        #         nextvalues = mb_values[t+1]
        #     delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
        #     mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        # mb_returns = mb_advs + mb_values

        #reset dones vector
        self.dones = [False]*len(self.dones)
        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, epinfos
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


# # Until all envs are done
# # env_dones = np.array([False] * self.env.num)
# while not all(self.dones):
#     # Given observations, get action value and neglopacs
#     # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
#
#     # do not act in envs that finished their test episode
#     dones = np.array(self.dones)
#     not_dones_mask = dones == False
#     states = self.states if self.states is None else self.states[not_dones_mask]
#     actions, values, self.states, neglogpacs = self.model.step(self.obs[not_dones_mask],
#                                                                S=states,
#                                                                M=dones[not_dones_mask].tolist())
#     mb_obs.append(self.obs.copy())


import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
from utils.logging import Logger
import numpy as np

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        anneal_time = args.t_max * args.epsilon_anneal_proportion

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, logger:Logger = None, print_log=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        if self.args.mask_unavailable_actions:
            masked_q_values[avail_actions == 0.0] = - \
                float("inf")  # should never be selected!
        
        if print_log:
            np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=np.inf)
            logger.console_logger.debug("Q-values".center(60, "-"))
            for agent_i in range(agent_inputs.shape[1]):
                logger.console_logger.debug(f"Agent{agent_i}: {agent_inputs[0][agent_i].detach().numpy()}")
            logger.console_logger.debug("Available actions".center(60, "-"))
            for agent_i in range(avail_actions.shape[1]):
                logger.console_logger.debug(f"Agent{agent_i}: {avail_actions[0][agent_i].detach().numpy()}")

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + \
            (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

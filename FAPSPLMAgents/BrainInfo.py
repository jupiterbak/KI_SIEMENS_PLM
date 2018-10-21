class BrainInfo:
    def __init__(self, observation, state, memory=None, reward=None, agents=None, local_done=None,
                 last_actions_discrete=None, last_actions_continuous=None):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.observations = observation
        self.states = state
        self.memories = memory
        self.rewards = reward
        self.local_done = local_done
        self.agents = agents
        self.last_action_discrete = last_actions_discrete
        self.last_actions_continuous = last_actions_continuous

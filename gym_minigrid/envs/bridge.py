import itertools

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class BridgeEnv(MiniGridEnv):
    """
    Single-room square grid environment with lava and collectable apples
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 5
    }

    def __init__(
            self,
            width=5,
            height=5,
            agent_start_pos=None,
            agent_start_dir=3,
            max_steps=10,
            action_noise=0.,
            reward_noise=0.
    ):
        self.agent_start_pos = agent_start_pos or (width // 2, height - 2)
        self.agent_start_dir = agent_start_dir
        self.action_noise = action_noise
        self.reward_noise = reward_noise
        self.done = False

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 3 actions permitted: move_up_left, move_up, move_up_right
        self.action_space = spaces.Discrete(3)

    def _bridge_grid(self, width, height, obstacle_type=Lava):
        assert width % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # # Place a goal square in the bottom-right corner

        for y in range(height):
            self.grid.set(0, y, obstacle_type())
            self.grid.set(1, y, obstacle_type())
            self.grid.set(width - 2, y, obstacle_type())
            self.grid.set(width - 1, y, obstacle_type())
        for x in range(2, width-2):
            self.grid.set(x, 0, Wall())
            self.grid.set(x, 1, Goal())
            self.grid.set(x, height - 1, Wall())

    def _gen_grid(self, width, height):
        self.start_pos = self.agent_start_pos
        self.start_dir = self.agent_start_dir
        self._bridge_grid(width, height)
        self.mission = "cross the bridge"

    def step(self, action):
        if self.done:
            return MiniGridEnv.step(self, MiniGridEnv.Actions.done)

        action = {0: MiniGridEnv.Actions.move_up_left,
                  1: MiniGridEnv.Actions.move_up,
                  2: MiniGridEnv.Actions.move_up_right}[action]

        # Step
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Action noise
        if not done and self.np_random.rand() < self.action_noise:
            noise_direction = self.np_random.choice([MiniGridEnv.Actions.move_right,
                                                     MiniGridEnv.Actions.move_left])
            self.step_count -= 1
            obs, reward, done, info = MiniGridEnv.step(self, noise_direction)

        # Symmetric reward perturbation
        if self.np_random.rand() < self.reward_noise:
            reward = 1 - reward

        self.done = done
        return obs, reward, done, info

    def _reward(self):
        """
        Reward received upon goal collection
        """
        return 1


class BridgeEnv9x9(BridgeEnv):
    def __init__(self):
        super().__init__(width=5, height=9, agent_start_pos=None, )


class BridgeEnvStochastic9x9(BridgeEnv):
    def __init__(self):
        super().__init__(width=7, height=9, agent_start_pos=None, action_noise=0.50)


register(
    id='MiniGrid-Bridge-v0',
    entry_point='gym_minigrid.envs:BridgeEnv9x9'
)
register(
    id='MiniGrid-Bridge-Stochastic-v0',
    entry_point='gym_minigrid.envs:BridgeEnvStochastic9x9'
)
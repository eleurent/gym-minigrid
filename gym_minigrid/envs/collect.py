from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class CollectEnv(MiniGridEnv):
    """
    Single-room square grid environment with lava and collectable apples
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second': 5
    }

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            n_goals=4,
            max_steps=20
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.n_goals = int(n_goals)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.move_up - self.actions.move_right + 1)
        self.action_offset = self.actions.move_right

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place goals
        for _ in range(self.n_goals):
            self.place_obj(Goal(), max_tries=100)

        # Place lava
        for _ in range(self.n_goals):
            self.place_obj(Lava(), max_tries=100)

        self.mission = "collect the green goals while avoiding the lava"

    def step(self, action):
        action = action + self.action_offset
        obs, reward, done, info = MiniGridEnv.step(self, action)

        cell = self.grid.get(*self.agent_pos)
        if cell is not None and cell.type == 'goal':
            self.grid.set(*self.agent_pos, None)  # Goals are collectable...
            done = False                          # ...and not terminal

        return obs, reward, done, info

    def _reward(self):
        """
        Compute the reward to be given upon goal collection
        """
        return 1


class CollectEnv6x6(CollectEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_goals=4)


register(
    id='MiniGrid-Collect-6x6-v0',
    entry_point='gym_minigrid.envs:CollectEnv6x6'
)

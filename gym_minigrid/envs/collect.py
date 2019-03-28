import itertools

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
            num_crossings=6,
            max_steps=20
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.n_goals = int(n_goals)
        self.num_crossings = int(num_crossings)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        self.action_offset = 0*self.actions.move_right

    def _crossing_grid(self, width, height, num_crossings=1, obstacle_type=Lava):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itertools.chain(
            itertools.product(range(1, width - 1), rivers_h),
            itertools.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.grid.set(i, j, obstacle_type())

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

    def _gen_grid(self, width, height):
        self._crossing_grid(width, height, self.num_crossings)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place goals
        for _ in range(self.n_goals):
            self.place_obj(Goal(), max_tries=100)

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


class CollectEnv9x9(CollectEnv):
    def __init__(self):
        super().__init__(size=9, agent_start_pos=None, n_goals=15, num_crossings=5)


register(
    id='MiniGrid-Collect-9x9-v0',
    entry_point='gym_minigrid.envs:CollectEnv9x9'
)

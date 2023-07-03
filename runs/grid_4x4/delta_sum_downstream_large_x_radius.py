from utils.rewards import DeltaSumRewardView
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250],
            [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]),
    reward=DeltaSumRewardView,
    run_name="grid_4x4_delta_downstream_sum_reward_large_x_radius",
    run_id=4,
    reward_radius=(0, 2),
)

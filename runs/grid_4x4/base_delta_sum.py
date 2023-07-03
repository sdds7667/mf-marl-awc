from utils.rewards import DeltaSumOutputReward
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250, 0, 750, 1500, 2250],
            [0, 0, 0, 0, 750, 750, 750, 750, 1500, 1500, 1500, 1500, 2250, 2250, 2250, 2250]),
    reward=DeltaSumOutputReward,
    run_name="grid_4x4_delta_sum_reward",
    run_id=5
)

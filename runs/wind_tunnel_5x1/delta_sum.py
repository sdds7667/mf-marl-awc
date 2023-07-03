
from utils.rewards import DeltaSumOutputReward
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500, 2250, 3000], [0, 0, 0, 0, 0]),
    reward=DeltaSumOutputReward,
    run_name="wt_5x1_delta_sum_reward",
    run_id=4
)

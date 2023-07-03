from utils.rewards import TotalPowerOutputReward
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500], [0, 0, 0]),
    reward=TotalPowerOutputReward,
    run_name="wt_3x1_sum_reward",
    run_id=1
)

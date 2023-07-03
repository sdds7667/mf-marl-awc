from utils.rewards import DeltaSumOutputReward, DeltaSumRewardView
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500, 2250, 3000], [0, 0, 0, 0, 0]),
    reward=DeltaSumRewardView,
    run_name="wt_5x1_delta_sum_reward_limited_view",
    run_id=1,
    radius=(0, 1),
    reward_radius=(0, 2),
)

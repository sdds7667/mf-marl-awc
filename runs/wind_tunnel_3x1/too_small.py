from utils.rewards import DeltaSumOutputReward, DeltaSumRewardView
from utils.run_config import RunConfig

config = RunConfig(
    layout=([0, 750, 1500], [0, 0, 0]),
    reward=DeltaSumRewardView,
    run_name="wt_3x1_too_small",
    run_id=1,
    radius=(0, 0),
    reward_radius=(0, 2),
)

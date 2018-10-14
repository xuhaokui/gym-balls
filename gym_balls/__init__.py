import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BallChaseRandomBall-v0',
    entry_point='gym_balls.envs:BallChaseRandomBallEnv',
    timestep_limit=2000,
)



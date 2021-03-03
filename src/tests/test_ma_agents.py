import random
from copy import copy

from src.common.observation_utils import flip_observation_horizontally


def test_flip_obs_horiz():
    obs = [random.random() for _ in range(4)] + [random.randint(0,1) for _ in range(6)]
    orig_obs = copy(obs)
    flipped = flip_observation_horizontally(obs)

    assert orig_obs == obs
    assert flipped[0] == orig_obs[0]
    assert flipped[2] == orig_obs[2]

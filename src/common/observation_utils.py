import numpy as np
from ma_gym.envs.pong_duel import pong_duel


def flip_observation_horizontally(obs, image_observations):
    if image_observations:
        reversed_obs = np.flip(obs, axis=1)
    else:
        # Ball directions should be ['NW', 'W', 'SW', 'SE', 'E', 'NE']
        # Consequently, when we reverse ball direction array, we flip horizontally
        agent_pos, ball_pos, ball_dir, opponent_pos = obs[:2], obs[2:4], obs[4:10], obs[10:]

        # The range is [0,1]. However, due to the way the observations are calculated, a position all the way to the left will be 0/30
        # and a position all the way to the right 29/30. Therefore, doing 1 - position is not enough, instead we have to offset by 1/30 in order
        # for everything to be exactly mirrored
        offset = 1 / 30
        # Perform the flipping
        agent_pos[1] = 1 - agent_pos[1] - offset
        ball_pos[1] = 1 - ball_pos[1] - offset
        ball_dir = np.flip(ball_dir)
        opponent_pos[1] = 1 - opponent_pos[1] - offset

        # Merge into single array
        reversed_obs = np.concatenate([agent_pos, ball_pos, ball_dir, opponent_pos])
    return reversed_obs


def is_ball_moving_towards_player(obs, player: str):
    ball_dir_one_hot = obs[4:10]
    ball_dir_i = np.argmax(ball_dir_one_hot)
    ball_dir = pong_duel.BALL_DIRECTIONS[ball_dir_i]
    if player == 'p1':
        if ball_dir in ['W', 'NW', 'SW']:
            return True
        else:
            return False
    elif player == 'p2':
        if ball_dir in ['E', 'NE', 'SE']:
            return True
        else:
            return False
    else:
        raise ValueError(f"Argument 'player' can be either 'p1' or 'p2', was {player}")

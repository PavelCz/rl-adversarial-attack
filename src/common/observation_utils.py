import numpy as np


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
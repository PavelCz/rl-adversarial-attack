import numpy as np
from ma_gym.envs.pong_duel import pong_duel

def approximate_step_pong_duel(env, current_state, action_p1, action_p2):
    p1_pos, ball_pos, ball_dir_one_hot, p2_pos = current_state[:2], current_state[2:4], current_state[4:10], current_state[10:]

    height, width = env.unwrapped._grid_shape  # Should be 40, 30
    move_amount_vert = 1 / height
    move_amount_horiz = 1 / width

    p1_pos = _move_agent(p1_pos, action_p1, move_amount_vert)
    p2_pos = _move_agent(p2_pos, action_p2, move_amount_vert)

    # Approximate the ball position
    # Ignore hitting other objects for now
    ball_dir_i = np.argmax(ball_dir_one_hot, axis=0)
    ball_dir = pong_duel.BALL_DIRECTIONS[ball_dir_i]

    if ball_dir == 'E':
        ball_pos[1] += move_amount_horiz
    elif ball_dir == 'W':
        ball_pos[1] -= move_amount_horiz
    elif ball_dir == 'NE':
        ball_pos[0] -= move_amount_vert
        ball_pos[1] += move_amount_horiz
    elif ball_dir == 'NW':
        ball_pos[0] -= move_amount_vert
        ball_pos[1] -= move_amount_horiz
    elif ball_dir == 'SE':
        ball_pos[0] += move_amount_vert
        ball_pos[1] += move_amount_horiz
    elif ball_dir == 'SW':
        ball_pos[0] += move_amount_vert
        ball_pos[1] -= move_amount_horiz
    else:
        raise Exception('Error in ball position')

    new_state = np.concatenate([p1_pos, ball_pos, ball_dir_one_hot, p2_pos])
    return new_state


def _move_agent(pos, action, move_amount):
    if action == 0:  # Don't move
        pass
    elif action == 1:  # Move up
        pos[0] -= move_amount
    elif action == 2:  # Move down
        pos[0] += move_amount
    else:
        raise Exception('Illegal action')
    return pos

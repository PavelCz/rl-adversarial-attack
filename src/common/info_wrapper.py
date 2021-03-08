import gym

from src.common.observation_utils import is_ball_moving_towards_player


class InfoWrapper(gym.Wrapper):
    """
    This is a wrapper for ma_gym environments. Works with pong_duel environment.
    Provides additional information in the info dict that is returned every step
    """

    def __init__(self, env: gym.Env):
        super(InfoWrapper, self).__init__(env)
        self.prev_targeted_player = None
        self.internal_info = {
            'p1_ball_hits': 0,
            'p2_ball_hits': 0,
            'ball_moved_towards_p1': 0,
            'ball_moved_towards_p2': 0
        }

    def step(self, actions):
        """

        """
        observations, rewards, dones, info = self.env.step(actions)
        [main_obs, opponent_obs] = observations
        targeted_player = 'p1' if is_ball_moving_towards_player(main_obs, 'p1') else 'p2'
        if self.prev_targeted_player is None:  # At the start of the game when the ball appears the first time
            self.internal_info['ball_moved_towards_' + targeted_player] += 1
            self.prev_targeted_player = targeted_player
        elif targeted_player != self.prev_targeted_player:
            # The player the ball is moving towards changed
            # This means the player must have hit the ball
            self.internal_info[self.prev_targeted_player + '_ball_hits'] += 1
            self.internal_info['ball_moved_towards_' + targeted_player] += 1
        else:
            # The ball is still moving in the same direction as in previous step -> nothing changes
            pass

        # Always update this
        self.prev_targeted_player = targeted_player

        # Apply internal infos to info dict
        self._add_infos(info)

        # Return the same data as the wrapped env with the only changes in the info dict
        return observations, rewards, dones, info

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        self.prev_targeted_player = None
        self.internal_info = {
            'p1_ball_hits': 0,
            'p2_ball_hits': 0,
            'ball_moved_towards_p1': 0,
            'ball_moved_towards_p2': 0
        }
        return obs

    def _add_infos(self, info_dict):
        for key in self.internal_info:
            info_dict[key] = self.internal_info[key]
# Adversarial Attacks on Reinforcement Learning Agents Trained with Self-Play

Project for the course *Advanced Deep Learning for Robotics*

Group `tum-adlr-ws20-03`

## Requirements

Some requirements only work with python 3.7

```
conda create -n adlr python==3.7
conda activate adlr
```

Install pip libraries from requirements file:

```
pip install -r requirements.txt
```

### Install PyTorch

```
conda install pytorch torchvision torchaudio -c pytorch
```


### Install multi-agent gym environment

``` bash
# At the same level as the project repository do:
git clone https://github.com/koulanurag/ma-gym.git
cd ma-gym
pip install -e .
```

## Running Regular Self-Play

Python scripts in the `src/scripts` folder to reproduce training and evaluation.
Settings for these scripts can be adjusted inside the scripts.

- `src/scripts/train_selfplay`
    - Training run for regular self-play
- `src/scripts/train_selfplay_adversarial_policy`
    - Train an adversarial policy that trains against a specified fixed victim
- `src/scripts/evaluate_model`
    - Evaluate trained agents by playing Pong
- `src/scripts/evaluate_observation_attack`
    - Evaluate agent trained with regular self-play against FGSM
- `src/scripts/render_match`
    - Show a Pong match with the specified trained agents
    

## Train Neural Fictitious Self-Play (NFSP)

Run `python main.py --env 'PongDuel-v0' --evaluate --render` to simply evaluate the trained agents playing against each other.

Run `python main.py --env 'PongDuel-v0' --evaluate --render --fgsm p1 --plot_fgsm` to attack agent 1 with Fast Gradient Sign Method(FGSM) and plot the perturbed state.

The original observation state of the agent from ma-gym is a 10 dimensional vector encoding the position and direction of the ball and two paddles, to change the observation state to an image observation run `python main.py --env 'PongDuel-v0' --evaluate --render --obs_img both`.

Run `python main.py --env 'PongDuel-v0'` to reproduce the training.

## Code Structure

- `models`: Trained models that were created as part of the project
- `src`: Python Source code root directory
    - `agents`
        - Implemented all-purpose agents
    - `attacks`
        - Code for both FGSM observation-based attack
        - Adversarial policy agent
        - Contains unsuccessful white-box adversarial policies
    - `common`
        - Common modules
    - `scripts`
        - Runnable python scripts for training, evaluating and visualizing
    - `selfplay`
        - Module containing code specific to self-play implementation
    - `tests`
        - Some tests

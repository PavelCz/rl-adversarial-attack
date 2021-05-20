# Adversarial Attacks on Reinforcement Learning Agents Trained with Self-Play in a Low-Dimensional Pong Environment

Attack using FGSM (perturbation in observation space), attack using adversarial policies

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
    

## Running Neural Fictitious Self-Play (NFSP)

- `main`
    - Main script for running NFSP agents
- `src/scripts/train`
    - Training run for NFSP
- `src/scripts/test`
    - Evaluate trained NFSP agents

Run `python main.py --env 'PongDuel-v0'` with built-in arguments to reproduce our training and testing results. Use `--obs_opp both` and `--obs_img both` to choose between feature-based and image-based observation for the agent, respectively. Use `--evaluate` to test trained agents. Use `--render` to visualize agents playing Pong. For more details please run `python main.py -h`. We give some examples about how to run the script that you may want to try out.
 
- `python main.py --env 'PongDuel-v0' --obs_opp both` 
    - Train NFSP agents with feature-based observation
- `python main.py --env 'PongDuel-v0' --obs_img both --evaluate --render --fgsm p1 --plot_fgsm`
    - Attack the left-hand side trained image-based NFSP agent using FGSM and plot the perturbed observation
- `python main.py --env 'PongDuel-v0' --obs_opp both --fgsm_training`
    - Adversarial training on NFSP

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

## Acknowledgements

- Pytorch implementation of NFSP is based on [pytorch-nfsp](https://github.com/younggyoseo/pytorch-nfsp) 

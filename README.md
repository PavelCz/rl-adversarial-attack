# Adversarial Attacks on Reinforcement Learning Agents Trained with Self-Play
Project for the course *Advanced Deep Learning for Robotics*

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

## Train Neural Fictitious Self-Play(NFSP)

Run `python main.py --env 'PongDuel-v0' --evaluate --render` to simply evaluate the trained agents playing against each other.

Run `python main.py --env 'PongDuel-v0' --evaluate --render --fgsm p1 --plot_fgsm` to attack agent 1 with Fast Gradient Sign Method(FGSM) and plot the perturbed state.

The original observation state of the agent from ma-gym is a 10 dimensional vector encoding the position and direction of the ball and two paddles, to change the observation state to an image observation run `python main.py --env 'PongDuel-v0' --evaluate --render --obs_img both`.

Run `python main.py --env 'PongDuel-v0'` to reproduce the training.


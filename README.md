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

### Install spinningup

At the same level as the project repository do:

``` bash
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```

### Install multi-agent gym environment

``` bash
# At the same level as the project repository do:
git clone https://github.com/koulanurag/ma-gym.git
cd ma-gym
pip install -e .
```
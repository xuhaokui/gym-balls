# Gym-Balls

[TOC]

Multiple balls chasing environment for both single agent and multiple agents. 

## Installations

### Pre-configuration

We will use the conda to build the virtual environment. 

```bash
conda create -n tf3 python=3.6 tensorflow-gpu
# conda create -n tf3 python=3.6 tensorflow (without GPU).
source activate tf3
# General Requirement
sudo apt-get install gcc libosmesa6-dev libgl1-mesa-dev libopenmpi-dev
# General Gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
# Mujoco-py
# Download the MuJoCo version 1.50 binaries for Linux, OSX, or Windows.
# Unzip the downloaded mjpro150 directory into ~/.mujoco/mjpro150, and place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt.
# add
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yd/.mujoco/mjpro150/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-387 # may need to change version
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so # may not necessary
# to .bashrc
git clone https://github.com/openai/mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
python setup.py install
# Baseline
pip install baselines
```

### Install the package

```bash
cd gym-balls
pip install -e .
```

### First example

```bash
python -m gym_balls.render --alg=ppo2 --env=BallChaseRandomBall-v0 --num_timesteps=0 --load_path=./models/ppo2_1e8 --play
```

### To render a video

```bash
python -m gym_balls.render --alg=ppo2 --env=BallChaseRandomBall-v0 --num_timesteps=0 --load_path=./models/ppo2_1e8
```

### To train a new model

```bash
python -m gym_balls.run --alg=ppo2 --env=BallChaseRandomBall-v0 --num_timesteps=1e7 --num_env=10 --save_path=./models/(name)
```

## Environments

### BallChaseRandomBall

Single agent setting with the agent controlling one ball to chase the other ball with random movements.
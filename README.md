## Autonomous Robot Navigation in Cluttered Environments Using Deep Reinforcement Learning

This repository contains the code for the paper "Autonomous Robot Navigation in Cluttered Environments Using Deep Reinforcement Learning."

### Dependencies

First, install a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next, install the environment. This project is built on the RVO2 library for collision avoidance and the PyTorch library for deep reinforcement learning. To install the environment, run the following commands or visit the [RVO2](https://github.com/sybrenstuvel/Python-RVO2/) repository for more information.

```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git rvo2
cd rvo2
pip install Cython
python setup.py build
python setup.py install
```

### Training

To train the model, run the following command.

```bash
./run_training.sh
```
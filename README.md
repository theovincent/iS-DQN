# Implementation of iterated Shared Deep Q-Network (`iS-DQN`)

[![custom_badge](https://img.shields.io/badge/ICLR_Paper-📄-7fe395)](https://arxiv.org/pdf/2506.04398)
[![custom_badge](https://img.shields.io/badge/OpenReview-📖-e3d77f)](https://openreview.net/forum?id=ltcxS7JE0c)

## User installation
We recommend using Python 3.11.5. In the folder where the code is, create a Python virtual environment, activate it, update pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as:```pytest```

## Running experiments
The script `launch_job/atari/launch.sh` trains an iS-DQN (K=9) agent with the CNN architecture and LayerNorm on a local machine, on the game Asterix.
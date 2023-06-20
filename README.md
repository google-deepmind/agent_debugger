# Agent Debugger

This repository provides an implementation of our paper [Causal Analysis of Agent Behavior for AI Safety](https://arxiv.org/abs/2103.03938).

>As machine learning systems become more powerful they also become increasingly unpredictable and opaque.
Yet, finding human-understandable explanations of how they work is essential for their safe deployment.
This technical report illustrates a methodology for investigating the causal mechanisms that drive the behaviour of artificial agents.
Six use cases are covered, each addressing a typical question an analyst might ask about an agent.
In particular, we show that each question cannot be addressed by pure observation alone, but instead requires conducting experiments with systematically chosen manipulations so as to generate the correct causal evidence.

The main tool is the "Agent Debugger", which can be used to perform causal interventions on the environment to infer the causal model of an agent.
We currently only support the environment Pycoworld, a 2D gridworld based on the open source game engine [Pycolab](https://github.com/deepmind/pycolab).


## Installation

Clone the source code into a local directory:
```bash
git clone https://github.com/deepmind/agent_debugger.git
cd agent_debugger
```

`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).
Then, create and activate the conda environment:
```bash
conda create --name agent_debugger
conda activate agent_debugger
```

Install `pip` and use it to install all the dependencies:
```bash
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use (CUDA 12 in the example above).
Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.


## Usage

Before running any code, make sure to activate the conda environment and set the `PYTHONPATH`:
```bash
conda activate agent_debugger
export PYTHONPATH=$(pwd)
```

See the 'experiments' [notebook](https://colab.research.google.com/github/deepmind/agent_debugger/blob/master/colabs/experiments.ipynb) to reproduce the experiments of the paper.


## Citing this work

```bibtex
@article{deletang2021causal,
  author       = {Gr{\'{e}}goire Del{\'{e}}tang and
                  Jordi Grau{-}Moya and
                  Miljan Martic and
                  Tim Genewein and
                  Tom McGrath and
                  Vladimir Mikulik and
                  Markus Kunesch and
                  Shane Legg and
                  Pedro A. Ortega},
  title        = {Causal Analysis of Agent Behavior for {AI} Safety},
  journal      = {arXiv:2103.03938},
  year         = {2021},
}
```


## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

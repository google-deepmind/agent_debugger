# agent_debugger

This repository contains the code associated with the paper [Causal analysis of
agent behaviour for AI safety.](https://arxiv.org/abs/2103.03938) (Deletang
et al., 2021). The main tool is called
the "Agent Debugger", which allows to do some causal interventions on the
environment to infer the causal model of an agent.

The only supported environment yet is Pycoworld, a 2D gridworld based on the
open source game engine [Pycolab](https://github.com/deepmind/pycolab).

## Installation

`pip install -r requirements.txt`

## Usage

See the 'experiments' [notebook](https://colab.research.google.com/github/deepmind/agent_debugger/blob/master/colabs/experiments.ipynb) to reproduce the experiments of the paper.

## Citing this work

```bibtex
@misc{deletang2021causal,
  author    = {Gr{\'{e}}goire Del{\'{e}}tang and
               Jordi Grau{-}Moya and
               Miljan Martic and
               Tim Genewein and
               Tom McGrath and
               Vladimir Mikulik and
               Markus Kunesch and
               Shane Legg and
               Pedro A. Ortega},
  title     = {Causal Analysis of Agent Behavior for {AI} Safety},
  year      = {2021},
  eprint    = {arXiv:2103.03938},
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

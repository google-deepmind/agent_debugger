# Causal Analysis of Agent Behavior for AI Safety

<p align="center">
  <img src="https://raw.githubusercontent.com/deepmind/agent_debugger/master/overview.jpg" alt="Overview figure"/>
</p>

This repository provides an implementation of our paper [Causal Analysis of Agent Behavior for AI Safety](https://arxiv.org/abs/2103.03938).

>As machine learning systems become more powerful they also become increasingly unpredictable and opaque.
Yet, finding human-understandable explanations of how they work is essential for their safe deployment.
This technical report illustrates a methodology for investigating the causal mechanisms that drive the behaviour of artificial agents.
Six use cases are covered, each addressing a typical question an analyst might ask about an agent.
In particular, we show that each question cannot be addressed by pure observation alone, but instead requires conducting experiments with systematically chosen manipulations so as to generate the correct causal evidence.

The main tool is the "Agent Debugger", which can be used to perform causal interventions on the environment to infer the causal model of an agent.
We currently only support the environment Pycoworld, a 2D gridworld based on the open source game engine [Pycolab](https://github.com/deepmind/pycolab).


## Usage

To reproduce the experiments of the paper, run the [experiments notebook](https://colab.research.google.com/github/deepmind/agent_debugger/blob/master/colabs/experiments.ipynb).


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

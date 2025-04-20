<p align="center">
  <img align="center" src="assets/logo.png" width="380px" />
</p>
<p align="left">

<div align="center">

<b>ZeroSumEval:</b> <em>An extensible framework for evaluating LLMs using games! ⚔</em>

</div>

<div align="center">

![Version](https://img.shields.io/pypi/v/zero-sum-eval)
[![arXiv](https://img.shields.io/badge/arXiv-2504.12562-b31b1b)](https://arxiv.org/abs/2504.12562)
[![Demo arXiv](https://img.shields.io/badge/demo_arXiv-2503.10673-b31b1b)](https://arxiv.org/abs/2503.10673)
![Pypi Downloads](https://img.shields.io/pypi/dm/zero_sum_eval)
[![Discord](https://img.shields.io/discord/1348445678589968466)](https://discord.gg/7Dk6jYyk8H)
[![Python package](https://github.com/facebookresearch/ZeroSumEval/actions/workflows/tests.yml/badge.svg)](https://github.com/haidark/ZeroSumEval/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/license/apache-2-0)

</div>


<!-- omit in toc -->
## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Games](#games)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgements)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

ZeroSumEval is a dynamic evaluation benchmark for LLMs using competitive scenarios that scales with model capabilities (i.e. as models get better, the benchmark gets harder). Instead of fixed evaluation benchmarks or subjective judging criteria, ZeroSumEval uses multi-agent simulations with clear win conditions to pit models against each other.

The framework tests various model capabilities, including knowledge, reasoning, and planning. In addition, ZeroSumEval uses [DSPy](https://github.com/stanfordnlp/dspy) optimization to test the self-improvement capability of models and ensure the competition between models is fair.

The eval suite consists of a growing number of simulations, including text-based challenges, board games, and Capture The Flag (CTF) competitions.

<p align="center">
  <img align="center" src="assets/bar.png" width="800px" />
  <br>
  <em>Performance comparison of different LLMs across various games in ZeroSumEval (March 9, 2025)</em>
</p>

Key features:
- One-click evals on the existing suite of games
- Easily extendable abstractions for new game implementations
- Integration with DSPy for automated prompt optimization
- Comprehensive logging and analysis tools


## Project Structure

The project is organized as follows:

- `zero_sum_eval/`: Main package containing the core framework
  - `analysis/`: Modules for analyzing game performance and calculating ratings
  - `core/`: Core game-related components, including player and game state management
  - `games/`: Individual game implementations
  - `managers/`: Game and match management classes
  - `utils/`: Utility functions for logging, configuration, checkpointing, and type definitions
  - `main.py`: Entry point for running games and matches
- `data/`: Game-specific data and examples
- `configs/`: Configuration files for different games and scenarios

## Installation

1. Use `pip` to install ZeroSumEval:
   ```
   pip install zero-sum-eval
   ```

2. test installation:
   ```
   zseval --help
   ```

## Usage

It's possible to run a single game or a series of matches with or without a detailed config file.

### Running without a config file

single game:
```
zseval -g chess -p "white=openai/gpt-4o" "black=openai/gpt-4o"
```

pool of matches:
```
zseval --pool -g chess -p "white=openai/gpt-4o" "black=openai/gpt-4o"
```

### Running from a config file

single game:
```
zseval -c configs/chess.yaml
```

pool of matches:
```
zseval --pool -c configs/pool/chess.yaml
```

### Rating calculation
Add the ```--calculate_ratings``` flag to output ELO ratings for the models after a pool of matches:
```
zseval --pool -c configs/pool/chess.yaml --calculate_ratings
```

Or directly calculate the ratings from a given match pool log directory:
```
zseval --calculate_ratings --output_dir match_pool_log/
```

## Games

ZeroSumEval currently supports the following games:

1. Chess
2. Debate
3. Gandalf (Password Guessing)
4. Liar's Dice
5. Math Quiz
6. Poker (Simple Texas Hold'em)
7. PyJail (Capture The Flag)

Each game is implemented as a separate module in the `zero_sum_eval/games/` directory.

## Configuration

Game configurations are defined in YAML files located in the `configs/` directory. These files specify:

- Logging settings
- Manager settings
- Game parameters
- Player configurations
- LLM settings

<details>
<summary>Example Configuration (chess.yaml):</summary>

```yaml
logging:
  output_dir: ../output/chess_game
manager:
  args:
    max_player_attempts: 5
    max_rounds: 200
game:
  name: chess
  args:
    players:
      white:
        class: chess_player
        args:
          id: llama3.1 70b white
          actions:
            - name: MakeMove
              optimize: true
              metric: chess_move_validation_metric
              dataset: chess_dataset
              dataset_args:
                filename: ./data/chess/stockfish_examples.jsonl
                player_key: white
                num_examples: 10
          lm:
            model: openrouter/meta-llama/llama-3.3-70b-instruct
          optimizer: BootstrapFewshot
          optimizer_args:
            max_bootstrapped_demos: 1
          max_tries: 5
      black:
        class: chess_player
        args:
          id: llama3.3 70b black
          lm:
            model: openrouter/meta-llama/llama-3.3-70b-instruct
          max_tries: 5
```
</details>

## Acknowledgements

Many thanks to Hisham Alyahya, Yazeed Alnumay, Colton Ritchie, and M Saiful Bari for their active contributions to the project. Because the project moved repositories, we were unable to preserve the commit history of the original repository.

## Citation

If you use ZeroSumEval in your work, please cite the following papers:

### [Paper](https://arxiv.org/abs/2504.12562):
```bibtex
@misc{khan2025zerosumevalscalingllmevaluation,
      title={ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition},
      author={Haidar Khan and Hisham A. Alyahya and Yazeed Alnumay and M Saiful Bari and Bülent Yener},
      year={2025},
      eprint={2504.12562},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.12562},
}
```

### [Demo Paper](https://arxiv.org/abs/2503.10673):
```bibtex
@misc{alyahya2025zerosumevalextensibleframeworkscaling,
      title={ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition},
      author={Hisham A. Alyahya and Haidar Khan and Yazeed Alnumay and M Saiful Bari and Bülent Yener},
      year={2025},
      eprint={2503.10673},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.10673},
}
```

## Contributing

Contributions to ZeroSumEval are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md) and open a pull request or issue on the [GitHub repository](https://github.com/facebookresearch/ZeroSumEval).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

ZeroSumEval collects model outputs in its logs for analysis and evaluation purposes. Each model's outputs are subject to the terms and conditions of the model's license and should be used in accordance with those terms.

## Star History

<a href="https://www.star-history.com/#facebookresearch/ZeroSumEval&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=facebookresearch/ZeroSumEval&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=facebookresearch/ZeroSumEval&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=facebookresearch/ZeroSumEval&type=Date" />
 </picture>
</a>

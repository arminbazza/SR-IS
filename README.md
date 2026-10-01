# Efficient learning of predictive maps for flexible planning
This repository is the code accompanying ["Efficient Learning of Predictive Maps for Flexible Planning"](https://www.biorxiv.org/content/10.64898/2026.02.11.705395v2.abstract). 

**Abstract:** Cognitive maps enable flexible behavior by providing reusable internal representations of task structure. The successor representation, a predictive map that encodes expected future state occupancy, has been proposed as one way such maps might be computed in the brain, but its policy dependence severely limits flexible planning. Here we propose the successor representation with importance sampling, a model that combines temporal-difference learning with importance sampling to construct policy-independent predictive maps. The model learns the structure of the environment without being constrained by the agent's current decision policy, resulting in a more general learned representation that can be efficiently updated when the environment changes to enable rapid behavioral adaptation. We show that it outperforms existing models in planning tasks and provides a better account of the graded biases in human replanning that previous models could not explain. This work bridges theories of predictive maps with observed planning behavior and offers new insights into flexible decision-making in the brain.

## Introduction
Everything should be self-contained inside of this repo. If you have any troubles running the code or if you have any quesstions you can reach out to me via email or make a GitHub issue.

## Usage
### Conda Environment
I recommend creating a conda environment for usage with this repo. You can install the conda environment I used from the yml file I have provided. Installation time can vary, but should be around 1-2 minutes.
```bash
conda env create -f env.yml
```

The code has been tested on a MacOS (13.6.7) with Python (3.10.0), MATLAB (version R2023b), and Julia (version 1.11.5).

### RL Environments
Because we are using custom gym environments, you need to install them locally in order for gymansium to recognize them. To install the environemnts, just run:
```bash
pip install -e gym-env
```

### Code synposis
The different models I tested can be found in `src/models.py`, the main models of interest for most readers are probably the `SR_IS` and `SR_IS_NHB` models. Note that there are two versions because one is defined to work on the tabular environments I constructed in `gym-env` while the other is designed to work on tree like environments that were used in ["Momennejad et al."](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OFdUAJwAAAAJ&citation_for_view=OFdUAJwAAAAJ:Tyk-4Ss8FVUC).

I have different notebooks, each containing different simulations that I used to construct my figures. They should be relatively easy to parse through.

## Citation
If you find this code useful, please reference it in your paper:
```
@article{bazarjani2026efficient,
  title={Efficient learning of predictive maps for flexible planning},
  author={Bazarjani, Armin and Piray, Payam},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
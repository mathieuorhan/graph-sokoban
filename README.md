# Sokoban: Learning policies with GNN

Repo for the project of the 2019 MVA course "Graph in Machine Learning".

In this work, we address the problem of learning general policies with Graph
Neural Networks to solve the Sokoban environment, a difficult planning puzzle.
We build a graph representation for Sokoban levels, learn policies with a Deep Q-learning
approach and then investigate the transfer and generalization capacities enabled by
GNNs of the learned policies.


## Getting Started

This implementation uses python 3.6, pytorch, Cuda 10.1.

### Prerequisites

First, install the requirements necessary to run this code. 

```
pip install -r requirements.txt
```

Then, you need to install Pytorch Geometric by following the installation tutorial [here](https://pytorch-geometric.readthedocs.io/en/latest/).

### Usage

Before training a model, you will need to generate Sokoban levels in a .png format. We implemented 3 ways of generating levels:
* By using the `gym-sokoban` environment:
```
python data/generate_gym.py
```
* Generate dummy levels:
```
python data/generate_dummy.py --dir [SAVE_DIR] --width [WIDTH] --height [HEIGHT] --boxes [NB_BOXES] --levels [NB_LEVELS]
```
* Build custom levels (need TKinter):
```
python data/level_builder.py --dir [SAVE_DIR] --width [WIDTH] --height [HEIGHT] --size [CELL_SIZE]
```
After generating levels, you can train models:
```
python train.py #parameters
```

The parameters for training are the following:
* `--train_path`, **"levels/dummy/train"**, *train dir*
* `--test_path`, **"levels/dummy/test"**, *test dir*
* `--training_id`, **str(int(time.time()))**, *log folder name*
* `--logs`, **"./logs/"**, *Log folder*
* `--pretrained`, **False** ,  *Use pretrained model*
* `--weights_path`, **""**, *path for pretrained weigths*
* `--save_every`, **10**, *Save model weigths every xx epochs*
* `--gpu`, **0**, *Which GPU ID to use*
* `--cpu`, **False**, *Use CPU*
* `--epochs`, **500**, *Number of epochs*
* `--seed`, **123**, *seed*
* `--max_steps`, **25**, *Maximum steps in environment for the train phase*
* `--max_steps_eval`, **25**, *Maximum steps in environment for the test phase*
* `--batch_size`, **32**, *batch size*
* `--buffer_size`, **10000**, *Replay memory buffer size*
* `--target_update`, **100**, *Target network update*
* `--gamma`, **1.0**, *discounted factor*
* `--eps_max`, **1.0**, *Epsilon-greedy initial value*
* `--eps_min`, **0.1**, *Epsilon greedy final minimum value*
* `--eps_stop_step`, **100000**, *Number of steps where epsilon reaches its min value*
* `--hiddens`, **64**, *hidden units*
* `--num_message_passing`, **2**, *number of EdgeConv layers*
* `--lr`, **0.0005**, *Learning rate*

Several information about the training will be stored in the log directory, and you can monitor them on the notebook `notebooks/history_vizualization.ipynb`.

## Built With

* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) - The Graph NN library used.

## Authors

* **Mathieu Orhan** - [mathieuorhan](https://github.com/mathieuorhan)
* **Bastien Déchamps** - [Bast24](https://github.com/Bast24)

## Bibliography

* [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
* [Learning combinatorial optimization
algorithms over graphs](https://arxiv.org/abs/1704.01665)
* [https://github.com/mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban)
* [Dynamic graph cnn for learning on point clouds](https://arxiv.org/abs/1801.07829)

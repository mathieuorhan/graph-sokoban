import torch.nn.functional as F

import options
from rl.qlearning_trainer import QLearningTrainer
from data.embedding import (
    MinimalEmbedding,
    NoWallsEmbedding,
    NoWallsV2Embedding,
    DirectionalEmbedding,
    PositionalEmbedding,
)

opt = options.parse_options()
opt.embedding = PositionalEmbedding()
opt.unet_hidden_channels = 64
opt.unet_depth = 3
opt.unet_pool_ratios = 0.5
opt.unet_sum_res = False
opt.unet_act = F.relu
trainer = QLearningTrainer(opt)

for _ in range(opt.epochs):
    trainer.train_one_epoch()
    trainer.eval_one_epoch()


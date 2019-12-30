import torch.nn.functional as F

import options
import logger
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

# Setup logger
logger.setup_logger(opt.logs)

trainer = QLearningTrainer(opt)

for _ in range(opt.epochs):
    trainer.train_one_epoch()
    trainer.eval_one_epoch()

# Plot an episode
trainer.render_one_episode(0)

import torch.nn.functional as F
import time

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
# opt.unet_hidden_channels = 256
# opt.unet_depth = 10
# opt.unet_pool_ratios = 0.5
# opt.unet_sum_res = False
# opt.unet_act = F.relu
opt.training_id = str(int(time.time()))

# Setup logger
logger.setup_logger(opt.logs, training_id=opt.training_id)

trainer = QLearningTrainer(opt)

for epoch in range(opt.epochs):
    trainer.train_one_epoch()
    trainer.eval_one_epoch()

    if opt.render and epoch % opt.render_every == 0:
        trainer.render_one_episode(0)

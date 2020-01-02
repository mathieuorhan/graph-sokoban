import torch
import torch.nn.functional as F
import time

import options
import logger
from model.graph_centered import GraphCenteredNet
from rl.qlearning_trainer_gc import QLearningGraphCenteredTrainer
from data.embedding import DirectionalPositionalEmbedding

opt = options.parse_options()
opt.embedding = DirectionalPositionalEmbedding()
opt.training_id = str(int(time.time()))

# Setup logger
logger.setup_logger(opt.logs, training_id=opt.training_id)

trainer = QLearningGraphCenteredTrainer(opt)
for epoch in range(opt.epochs):
    trainer.train_one_epoch()
    trainer.eval_one_epoch()


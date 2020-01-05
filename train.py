import torch
import torch.nn.functional as F
import time
import pickle
import os

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
history = dict()
trainer = QLearningGraphCenteredTrainer(opt)

# Training loop
for epoch in range(opt.epochs):
    epoch_info = trainer.train_one_epoch()
    # monitor the information about training
    for info in epoch_info:
        if info not in history:
            history[info] = [epoch_info[info]]
        else:
            history[info].append(epoch_info[info])

    # Evaluation
    trainer.eval_one_epoch()

    # Save weights
    trainer.save_model()

# Saving history in the logs
with open(os.path.join(opt.logs, opt.training_id, "history.pkl"), "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

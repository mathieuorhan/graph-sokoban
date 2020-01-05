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
history_train = dict()
history_eval = dict()
trainer = QLearningGraphCenteredTrainer(opt)

# Training loop
for epoch in range(opt.epochs):
    # Train
    train_epoch_info = trainer.train_one_epoch()
    # monitor the information about training
    for info in train_epoch_info:
        if info not in history_train:
            history_train[info] = [train_epoch_info[info]]
        else:
            history_train[info].append(train_epoch_info[info])

    # Evaluate
    eval_epoch_info = trainer.eval_one_epoch()
    # monitor the information about training
    for info in eval_epoch_info:
        if info not in history_eval:
            history_eval[info] = [eval_epoch_info[info]]
        else:
            history_eval[info].append(eval_epoch_info[info])

    if opt.render and epoch % opt.render_every == 0:
        trainer.render_one_episode(0)

    # Save weights
    trainer.save_model()

# Saving history in the logs
with open(os.path.join(opt.logs, opt.training_id, "train_history.pkl"), "wb") as f:
    pickle.dump(history_train, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(opt.logs, opt.training_id, "eval_history.pkl"), "wb") as f:
    pickle.dump(history_eval, f, pickle.HIGHEST_PROTOCOL)

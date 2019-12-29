class AnnealingScheduler:
    """Annealing (epsilon) scheduler"""

    def __init__(self, epsilon_max, epsilon_min, stopping_step):
        self.current_step = 0
        self.stopping_step = stopping_step
        self.epsilon = epsilon_max
        self.decay = (epsilon_max - epsilon_min) / (stopping_step)

    def step(self):
        if self.current_step < self.stopping_step:
            self.epsilon -= self.decay
        self.current_step += 1
        return self.epsilon

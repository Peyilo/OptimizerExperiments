class EMA(object):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.ema = 0
        self.count = 0

    def update(self, value):
        self.ema = (1 - self.beta) * self.ema + self.beta * value
        self.count += 1
        return self.ema / (1 - self.beta ** self.count)

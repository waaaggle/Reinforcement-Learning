from samples.basic_env import BasicEnviroment

class NetworkEnv(BasicEnviroment):
    def __init__(self):
        self.s = 0.0
        self.a = 0
        self.r = 0
        self.s_next = 0
        self.done = 0
        self.info = 0

    def reset(self):
        self.s = 0
        self.a = 0
        self.r = 0
        self.s_next = 0
        self.done = 0
        self.info = 0

    def step(self):
        return self.s,self.a,self.r,self.s_next,self.done,self.info
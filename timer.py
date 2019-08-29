import time
import os

def sec2hms(sec):
    t = sec
    s, t = t%60, t//60
    m, t = t%60, t//60
    h, d = t%24, t//24
    if d > 0: return "{:.0f}d {:.0f}h {:.0f}m {:.0f}s".format(d,h,m,s)
    if h > 0: return "{:.0f}h {:.0f}m {:.0f}s".format(h,m,s)
    if m > 0: return "{:.0f}m {:.0f}s".format(m,s)
    return "{:.02f}s".format(s)

class Clock:
    def __init__(self, n_steps=None):
        self.tic()
        self.n_steps = n_steps
    def tic(self):
        self.start_time = time.time()
    def toc(self, step=None):
        cost = time.time() - self.start_time
        if step is None or self.n_steps is None: 
            return sec2hms(cost)
        else:
            step += 1
            return sec2hms(cost), sec2hms(cost*(self.n_steps-step)/step)

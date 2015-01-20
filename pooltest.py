from multiprocessing import Pool
from numpy import linspace, mean, array, ones

class agent:
    def __init__(self, alpha=1, delta=0.08):
    	self.alpha = array([alpha for i in range(10)], dtype=float)
    	self.delta = array([delta for i in range(10)], dtype=float)

    def squarea(self,a):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        self.alpha = self.alpha*a


    def squared(self,a):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        self.delta = self.delta*a

# if __name__ == '__main__':
#     p = Pool(5)
#     print(p.map(f, [1, 2, 3]))


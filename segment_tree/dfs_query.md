线段树模板题

```python
# 给定 return 序列，找第 n 大的 drawdown，如果不存在，返回 drawdown 个数

class Result:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def updateMaxDD(self, res2):
        if res2.maxDD < self.maxDD:
            self.maxDD = res2.maxDD
            self.x = res2.x
            self.y = res2.y

class Node:
    def __init__(self, st, ed):
        
        self.st = st
        self.ed = ed
        self.mid = (st + ed) // 2
        self._lc = None
        self._rc = None
        
        self.maxDD = None
        self.x = None
        self.y = None
        self.minIdx = None
        self.maxIdx = None
        
    @property
    def lc(self):
        self._lc = self._lc or Node(self.st, self.mid)
        return self._lc
        
    @property
    def rc(self):
        self._rc = self._rc or Node(self.mid + 1, self.ed)
        return self._rc
    
    def query(self, l, r, prices):
        if self.st == l and self.ed == r and self.maxDD is not None:
            return Result(
                x=self.x,
                y=self.y,
                maxDD=self.maxDD,
                minIdx=self.minIdx,
                maxIdx=self.maxIdx
            )
        
        if self.st == self.ed:
            self.x = self.st
            self.y = self.st
            self.maxDD = 0
            self.minIdx = self.st
            self.maxIdx = self.st
            return self.query(l, r, prices)
        
        if r <= self.mid:
            return self.lc.query(l, r, prices)
        if l >= self.mid + 1:
            return self.rc.query(l, r, prices)
        
        res1 = self.lc.query(l, self.mid, prices)
        res2 = self.rc.query(self.mid + 1, r, prices)
        res = Result(x=res1.maxIdx, y=res2.minIdx, maxDD=(prices[res2.minIdx] - prices[res1.maxIdx]) / prices[res1.maxIdx],
                     minIdx = res1.minIdx if prices[res1.minIdx] <= prices[res2.minIdx] else res2.minIdx,
                     maxIdx = res1.maxIdx if prices[res1.maxIdx] >= prices[res2.maxIdx] else res2.maxIdx)
        res.updateMaxDD(res1)
        res.updateMaxDD(res2)
        
        return res
    
def maxdd(rets=[-0.09,0.5, -0.1, -0.1, 0.01, 1,1,1], k=1):
    prices = [1]
    for ret in rets:
        prices.append(prices[-1] * (1 + ret))
    
    N = len(prices)
    tree = Node(0, N - 1)

    maxDDs = []
    
    def dfs(l, r):
        res = tree.query(l, r, prices)
        if res.maxDD >= 0:
            return
        maxDDs.append(res.maxDD)
        if res.x - 1 >= l:
            dfs(l, res.x - 1)
        if r >= res.y + 1:
            dfs(res.y + 1, r)
    
    dfs(0, N - 1)
    if len(maxDDs) >= k:
        import heapq
        return heapq.nsmallest(k, maxDDs)[-1]
    return len(maxDDs)

maxdd()

```




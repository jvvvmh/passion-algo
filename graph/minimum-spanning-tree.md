## 最小生成树 MST

切分定理：在一幅连通加权无向图中，给定任意的切分，如果有一条横切边的权值严格小于所有其他横切边，则这条边必然属于图的最小生成树中的一条边。

[TOC]

### Kruskal 算法

边排序，只要不形成环（并查集），就加入，直到有 n - 1 条边为止。

### 力扣1584. 连接所有点的最小费用

给定一些点的坐标，cost 是曼哈顿距离，求连通所有点的最小费用。

```python
import heapq
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:

        # Kruskal：找最小边，如果不连通，则加上这个边
        n = len(points)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                edges.append((dist, i, j))
        
        heapq.heapify(edges)
        cnt = 0
        res = 0
        root = list(range(n))
        rank = [1] * n

        def find(x):
            if root[x] != x:
                root[x] = find(root[x])
            return root[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX == rootY:
                return False
            if rank[rootX] > rank[rootY]:
                root[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                root[rootX] = rootY
            else:
                rank[rootX] += 1
                root[rootY] = rootX
            return True

        while cnt < n - 1:
            dist, x, y = heapq.heappop(edges)
            if union(x, y):
                cnt += 1
                res += dist

        return res
```

### Prim 算法

visited 和 unvisited。每次新加一个点到 visited 就更新横切边。在横切边中，选最小的加入，然后更新 visited。

- 力扣1584. 连接所有点的最小费用

```python
import heapq
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:

        # Prim: visited / unvisited 切边中最小的。
        # 每次多一个新 visited 的点，更新切边
        # 初始 visited = [0]

        n = len(points)
        def dist(i, j):
            return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
        graph = [[dist(i, j) for j in range(n)] for i in range(n)]
        
        lastVisit = 0
        res = 0
        q = []
        unvisited = set([i for i in range(1, n)])
        while unvisited:
            for j in unvisited:
                heapq.heappush(q, (graph[lastVisit][j], j))
            dist, lastVisit = heapq.heappop(q)
            while lastVisit not in unvisited:
                dist, lastVisit = heapq.heappop(q)
            unvisited.remove(lastVisit)
            res += dist
        return res
```




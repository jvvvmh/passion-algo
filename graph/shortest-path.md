## 最短路径

[TOC]

### Dijkstra 单源最短路径

「Dijkstra 算法」只能解决加权有向图的权重为非负数的「单源最短路径」问题。

<img src="images\Dijkstra-negative.PNG" alt="Dijkstra-negative" style="zoom:38%;" />



### 力扣 743. 网络延迟时间

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        k -= 1

        dist = [float('inf') for _ in range(n)]
        dist[k] = 0
        unvisited = set([i for i in range(n)])
        graph = defaultdict(list)
        for st, ed, w in times:
            graph[st - 1].append((ed - 1, w))

        for _ in range(n):
            min_dist = float('inf')
            best_node = None
            for node in unvisited: # 未访问顶点中，最短的距离
                if dist[node] < min_dist:
                    min_dist, best_node = dist[node], node
            if min_dist == float('inf'): # 无法更新，不连通
                return -1
            unvisited.remove(best_node)
            for y, w in graph[best_node]:
                dist[y] = min(dist[y], dist[best_node] + w)
        max_dist = max(dist)
        return max_dist if max_dist != float('inf') else -1
```

### Bellman-Ford 单源最短路径

「Bellman-Ford 算法」能解决加权有向图中包含权重为负数的「单源最短路径」问题。

**定理一：在一个有 N 个顶点的「非负权环图」中，两点之间的最短路径最多经过 N−1 条边。**

**定理二：「负权环」没有最短路径。**

最多经历1,2，...，N - 1条边的最短路径。

检测负权环：第N次更行是否能减小路径

BF 提升：每次循环遍历所有边，更新距离，直到有一次循环什么都不能提升为止。但无法解决最多 K 次更新的问题。

### SPFA 算法

针对一个无负权环的图。

「SPFA 算法」主要是通过「队列」来维护我们接下来要遍历边的起点，而不是「Bellman Ford」算法中的任意还没有遍历过的边。每次只有**当某个顶点的最短距离更新**之后，**且该顶点不在「队列」中**，我们就将该顶点加入到「队列」中。一直循环以上步骤，**直到「队列」为空**。



### 力扣 787. K 站中转内最便宜的航班

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist = [float('inf')] * n
        dist[src] = 0
        for _ in range(k + 1): # BF
            new_dist = [x for x in dist]
            for st, ed, w in flights:
                new_dist[ed] = min(new_dist[ed], dist[st] + w)
            dist = new_dist
        return -1 if dist[dst] == float('inf') else dist[dst]
```


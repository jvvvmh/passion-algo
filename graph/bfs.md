## BFS

1. 遍历图中所有顶点
2. 针对权重相等且均为正数的图，快速找出两点之间的最短路径

[TOC]

### 力扣 797. 所有可能的路径

 directed acyclic graph (**DAG**) 

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        ed = len(graph) - 1
        q = [[0]]
        res = []
        while q:
            head = q[0]
            q.pop(0)
            if head[-1] == ed:
                res.append(head)
                continue
            x = head[-1]
            for y in graph[x]:
                q.append(head + [y])
        return res
```


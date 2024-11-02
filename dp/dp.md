## DFS

遍历定点，遍历两点间的所有路径

[TOC]

### 力扣 797. 所有可能的路径

 directed acyclic graph (**DAG**) 

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res = []
        curr = []
        def dfs(x, ed, curr, res):
            if x == ed:
                res.append(curr.copy())
                return
            for y in graph[x]:
                curr.append(y)
                dfs(y, ed, curr, res)
                curr.pop()
        curr.append(0)
        dfs(0, len(graph) - 1, curr, res)
        return res
```


## 拓扑排序

[TOC]

队列维护入度为 0 的点，更新别人的入度，直到队列为空。

### 力扣 210. 课程表 II

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        in_degree = [0] * numCourses
        graph = defaultdict(list)
        for y, x in prerequisites:
            graph[x].append(y)
            in_degree[y] += 1
        q = []
        ans = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                q.append(i)
        while q:
            x = q.pop(0)
            ans.append(x)
            for y in graph[x]:
                in_degree[y] -= 1
                if in_degree[y] == 0:
                    q.append(y)
        if len(ans) < numCourses: # 环
            return []
        return ans
```


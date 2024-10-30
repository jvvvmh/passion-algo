# Graph

[TOC]

## 并查集

按秩合并 + 路径压缩

### 547. 省份数量

``isConnected[i][j]=1`` 表示城市相连

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        rank = [1] * n
        root = list(range(n))
        cnt = n

        def find(x):
            if root[x] == x:
                return x
            root[x] = find(root[x]) # 路径压缩
            return root[x]

        def union(x, y, cnt):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                cnt -= 1 # 每次合并，省份-1
                if rank[rootX] > rank[rootY]: # 按秩合并
                    root[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    root[rootX] = rootY
                else:
                    root[rootY] = rootX
                    rank[rootX] +=1
            return cnt

        for i in range(n):
            for j in range(i):
                if isConnected[i][j]:
                    cnt = union(i, j, cnt)
        return cnt
```


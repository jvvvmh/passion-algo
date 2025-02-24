双指针



[792. Number of Matching Subsequences](https://leetcode.cn/problems/number-of-matching-subsequences/)

```python
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        p = defaultdict(list)
        for i, w in enumerate(words):
            p[w[0]].append((i, 0))
        ans = 0
        for c in s:
            q = p[c]
            p[c] = []
            for i, j in q:
                j += 1
                if j == len(words[i]):
                    ans += 1
                else:
                    p[words[i][j]].append((i, j))
        return ans
 
```


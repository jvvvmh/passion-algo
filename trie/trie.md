Trie



[208. Implement Trie (Prefix Tree)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

```python
class Trie:

    def __init__(self):
        self.child = [None] * 26
        self.isEnd = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            idx = ord(ch) - ord("a")
            if not node.child[idx]:
                node.child[idx] = Trie()
            node = node.child[idx]
        node.isEnd = True

    def searchPrefix(self, prefix: str) -> "Trie":
        node = self
        for ch in prefix:
            idx = ord(ch) - ord("a")
            if node.child[idx] is None:
                return None
            node = node.child[idx]
        return node

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        node = self.searchPrefix(prefix)
        return node is not None
```



[14. Longest Common Prefix](https://leetcode.cn/problems/longest-common-prefix/)

```python
class Trie:
    def __init__(self):
        self.childs = [None] * 26
        self.cnt = 0

    def insert(self, word):
        node = self
        for ch in word:
            idx = ord(ch) - ord("a")
            if node.childs[idx] is None:
                node.childs[idx] = Trie()
            node = node.childs[idx]
            node.cnt += 1
    
    def searchPrefix(self, prefix, n):
        node = self
        res = 0
        for ch in prefix:
            idx = ord(ch) - ord("a")
            if node.childs[idx] is None:
                return res
            node = node.childs[idx]
            if node.cnt == n:
                res += 1
        return res

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        n = len(strs)
        node = Trie()
        for s in strs:
            node.insert(s)
        return strs[0][:node.searchPrefix(strs[0], n)]
        
```


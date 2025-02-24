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


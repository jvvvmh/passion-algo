# 单调栈 - 树

[TOC]

### [255. Verify Preorder Sequence in Binary Search Tree](https://leetcode.cn/problems/verify-preorder-sequence-in-binary-search-tree/)

考虑二叉搜索树的特点：

- 每个节点的左子树只包含小于当前节点的数；
- 每个节点的右子树只包含大于当前节点的数；
- 所有左子树和右子树自身必须也是二叉搜索树。

如果栈为空，或者当前值小于栈顶的值，则将当前值压入栈内。如果当前值大于栈顶的值，说明当前值是某个节点的右子节点的值，因此将栈内小于当前值的元素全部弹出，然后将当前值压入栈内。

**最后弹出的元素是当前值所在节点的父节点的元素**，即当前值所在节点是该父节点的右子节点，因此将最小值设为最后弹出的元素。根据二叉搜索树的特点，**在父节点的右子树中的任何节点的值都必须大于父节点的值**，即此时的最小值，如果在先序遍历序列中发现一个值小于或等于最小值，则该先序遍历序列不是二叉搜索树的正确先序遍历序列。

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        s = []
        lb = float('-inf')
        for num in preorder:
            if num <= lb:
                return False
            while s and s[-1] <= num:
                lb = s.pop()
            s.append(num)
        return True

class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        topIdx = -1
        lb = float('-inf')
        for num in preorder:
            if num <= lb:
                return False
            while topIdx != -1 and preorder[topIdx] <= num:
                lb = preorder[topIdx]
                topIdx -=1
            topIdx += 1
            preorder[topIdx] = num
        return True
```



```python

```




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
```

### [LCR 152. 验证二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

```python
class Solution:
    def verifyTreeOrder(self, postorder: List[int]) -> bool:
        # post order: left->right->me
        # reverse:    me -> right -> left
        # left pop the largers, the me gives an upper bound
        # if a new element comes, which belong to the left child, the element should be smaller than me
        
        s = []
        upperBound = float('inf')
        for i in range(len(postorder) - 1, -1, -1):
            if postorder[i] > upperBound:
                return False
            while s and postorder[i] < s[-1]:
                upperBound = s.pop()
            s.append(postorder[i])
        return True


```





### [654. Maximum Binary Tree](https://leetcode.cn/problems/maximum-binary-tree/)

array of distinct numbers. 取最大的作为root，把数组一分为二，然后左右数组recursive

1. BFS 最差情况数组单调，分割 O(N)，组内寻找最大 O(N)

2. 自己被visit的时候，比自己大的都visit过了。从方法1来看，左边最近的比自己大的，右边最近的比自己大的，较小者是自己的父节点（反证法）。但是需要存left/right数组。
3. 在栈上动态更新：维护一个单调递减，pop 左边比自己小的，我指向它。栈中左边比自己大的再指向自己。如果后来来了一个比自己大的，那么，它pop我，我指向它；我的栈中左边指向它。

<img src="images\max-tree.PNG" alt="tree" style="zoom: 50%;" >

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        # 5 [4] 2 1 (3)
        s = []
        nodes = [TreeNode(val=num) for num in nums]
        for i, num in enumerate(nums):
            while s and num > nums[s[-1]]:
                idx = s.pop()
                nodes[i].left = nodes[idx]
            if s:
                nodes[s[-1]].right = nodes[i]
            s.append(i)
        return nodes[s[0]]
```

### [1008. Construct Binary Search Tree from Preorder Traversal](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        nodes = [TreeNode(num) for i, num in enumerate(preorder)]
        s = [0]
        for i in range(1, len(preorder)):
            if s and preorder[i] < preorder[s[-1]]:
                nodes[s[-1]].left = nodes[i]
            rootIdx = -1
            while s and preorder[i] > preorder[s[-1]]:
                rootIdx = s.pop()
            if rootIdx != -1:
                nodes[rootIdx].right = nodes[i]
            s.append(i)
        return nodes[0]
```

### [1130. Minimum Cost Tree From Leaf Values](https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/)

<img src="images\tree-merge-cost.PNG" alt="tree" style="zoom: 50%;" >

合并cost是左右子树最大值的乘积。希望小的数字多合并几次。

4 3 2 1 从右边开始合并

1 2 3 4 从左边开始合并

8 5 (2 1) 6 -> 8 5 2 6 -> 8 5 6 -> 8 6

4 2 3 合并 2, 3 -> 2, 3

```python
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        ans = 0
        s = []
        for x in arr:
            while s and x >= s[-1]:
                y = s.pop()
                if not s or s[-1] > x: # 和自己合并, 新 value 是自己
                    ans += y * x
                    break
                else:
                    ans += s[-1] * y # 和栈中元素合并, 新 value 是栈中元素，还在栈中
            # 栈中元素已经比自己严格大了
            s.append(x)
        # s 中元素单调递减, 从后面开始 4 3 2 1
        while len(s) >= 2:
            x = s.pop()
            ans += s[-1] * x
        return ans
```


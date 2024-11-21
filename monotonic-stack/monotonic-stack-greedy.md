# 单调栈 - 贪心

[TOC]

### [316. Remove Duplicate Letters](https://leetcode.cn/problems/remove-duplicate-letters/)

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is **the smallest in lexicographical order** among all possible results.

**Example 1:**

```
Input: s = "bcabc"
Output: "abc"
```

**Example 2:**

```
Input: s = "cbacdcbc"
Output: "acdb"
```

```c++
class Solution {
public:
    string removeDuplicateLetters(string s) {
        vector<int> visited(26), cnt(26);
        for (char ch: s) {
            ++cnt[ch - 'a'];
        }
        string stk;
        for (char ch: s) {
            if (!visited[ch - 'a']) {
                while (!stk.empty() && stk.back() > ch) {
                    if (cnt[stk.back() - 'a'] > 0) {
                        visited[stk.back() - 'a'] = 0;
                        stk.pop_back();
                    } else {
                        break;
                    }
                }
                stk.push_back(ch);
                visited[ch - 'a'] = 1;
            }
            --cnt[ch - 'a'];
        }
        return stk;
    }
};
```

 

### [321. Create Maximum Number](https://leetcode.cn/problems/create-maximum-number/)

**Example 1:**

```
Input: nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5
Output: [9,8,6,5,3]
```

**Example 2:**

```
Input: nums1 = [6,7], nums2 = [6,0,4], k = 5
Output: [6,7,6,0,4]
```

```python
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        m, n = len(nums1), len(nums2)
        ans = [0] * k
        def selectK(x, k): # 单调递减 unless 不够
            res = []
            for i in range(len(x)):
                if not res or res[-1] >= x[i]:
                    res.append(x[i])
                else:
                    while res and res[-1] < x[i] and len(res) - 1 + len(x) - i >= k:
                        res.pop()
                    res.append(x[i])
            return res[:k]

        def merge(s1, s2): # 按字典序
            l1, l2 = len(s1), len(s2)
            if l1 == 0:
                return s2
            if l2 == 0:
                return s1
            totalL = l1 + l2
            res = []
            idx1, idx2 = 0, 0
            for _ in range(totalL):
                if cmp(s1, idx1, s2, idx2) > 0:
                    res.append(s1[idx1])
                    idx1 += 1
                else:
                    res.append(s2[idx2])
                    idx2 += 1
            return res

        def cmp(s1, idx1, s2, idx2):
            while idx1 < len(s1) and idx2 < len(s2):
                diff = s1[idx1] - s2[idx2]
                if diff:
                    return diff
                idx1 += 1
                idx2 += 1
            return idx1 < len(s1)
                
        for l1 in range(m + 1):
            l2 = k - l1
            if l2 < 0 or l2 > n:
                continue
            s1 = selectK(nums1, l1)
            s2 = selectK(nums2, l2)
            merged = merge(s1, s2)

            update = False
            for i in range(k):
                if merged[i] > ans[i]:
                    update = True
                    break
                elif merged[i] < ans[i]:
                    break
            if update:
                ans = merged
        return ans
```



### [402. Remove K Digits](https://leetcode.cn/problems/remove-k-digits/)

Given string num representing a non-negative integer `num`, and an integer `k`, return *the smallest possible integer after removing* `k` *digits from* `num`.

**Example 1:**

```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
```

**Example 2:**

```
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
```

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        # 2 2 3
        s = ""
        for ch in num:
            if not s or ch >= s[-1]:
                s += ch
            else:
                while s and ch < s[-1] and k > 0:
                    s = s[:-1]
                    k -= 1
                s += ch
                if len(s) == 1 and ch == '0':
                    s = ""
        if k >= len(s):
            s = ""
        elif k > 0:
            s = s[:-k]
        
        while s and s[0] == '0':
            s = s[1:]
        
        if len(s) == 0:
            s = '0'
        
        return s
```

### [456. 132 Pattern](https://leetcode.cn/problems/132-pattern/)

```python
from sortedcontainers import SortedList

class Solution:
    def find132pattern(self, nums: List[int]) -> bool:

        # O(n), max_k is a valid "2"
        n = len(nums)
        candidate_k = [nums[n - 1]]
        max_k = float("-inf")

        for i in range(n - 2, -1, -1):
            if nums[i] < max_k:
                return True
            while candidate_k and nums[i] > candidate_k[-1]:
                max_k = candidate_k[-1]
                candidate_k.pop()
            if nums[i] > max_k: # > max_k 但是 <= last candidate
                                # candidate 是 单调递减的
                candidate_k.append(nums[i])
        return False

 
        # multiset or SortedList nlog(n)
        if len(nums) < 3:
            return False
        leftMin = [nums[0]]
        for x in nums[1:]:
            leftMin.append(min(leftMin[-1], x))

        rightAll = SortedList(nums[-1:])
        for i in range(len(nums) - 2, 0, -1):
            if leftMin[i] < nums[i]:
                index = rightAll.bisect_right(leftMin[i])
                if index < len(rightAll) and rightAll[index] < nums[i]:
                    return True
            rightAll.add(nums[i])
        return False

 
```

### [503. Next Greater Element II](https://leetcode.cn/problems/next-greater-element-ii/)

循环数组中，下一个大于自己的数字

```c++
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int> res(n, -1);
        stack<int> s;
        for (int i = 0; i < n * 2 - 1; ++i) {
            while (!s.empty() && nums[s.top()] < nums[i % n]) {
                res[s.top()] = nums[i % n];
                s.pop();
            }
            s.push(i % n);
        }
        return res;
    }
};
```

### [581. Shortest Unsorted Continuous Subarray](https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/)

Given an integer array `nums`, you need to find one **continuous subarray** such that if you only sort this subarray in non-decreasing order, then the whole array will be sorted in non-decreasing order.

**Example 1:**

```
Input: nums = [2,6,4,8,10,9,15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
```

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # leftmost time to break inc
        # i.e. exists a number right to me but smaller than me
        # i.e. < rightMin
        leftIdx = -1
        rMin = float('inf')
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] > rMin:
                leftIdx = i
            else:
                rMin = nums[i]
        if leftIdx == -1:
            return 0
        # rightmost time to break inc
        # i.e. me < leftMax
        rightIdx = -1
        leftMax = float('-inf')
        for i in range(len(nums)):
            if nums[i] < leftMax:
                rightIdx = i
            else:
                leftMax = nums[i]
        return rightIdx - leftIdx + 1
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


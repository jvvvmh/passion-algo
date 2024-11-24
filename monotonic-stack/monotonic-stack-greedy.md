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

### [962. Maximum Width Ramp](https://leetcode.cn/problems/maximum-width-ramp/)

A **ramp** in an integer array `nums` is a pair `(i, j)` for which `i < j` and `nums[i] <= nums[j]`. The **width** of such a ramp is `j - i`.

Given an integer array `nums`, return *the maximum width of a **ramp** in* `nums`.

```python
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        # 排序: 之前最小idx和自己
        # arr = [(x, i) for i, x in enumerate(nums)]
        # arr = sorted(arr)
        prevMinIdx = float('inf')
        res = 0
        n = len(nums)
        for currIdx in sorted(range(n), key=nums.__getitem__):
            res = max(res, currIdx - prevMinIdx)
            prevMinIdx = min(prevMinIdx, currIdx)
        return res

        # 单调递减栈 (左边可能的)
        # 从右边开始，pop这个栈
        s = [0]
        for i in range(1, len(nums)):
            if nums[i] < nums[s[-1]]:
                s.append(i)
        res = 0
        for i in range(len(nums) - 1, -1, -1):
            while (s and i >= s[-1] and nums[i] >= nums[s[-1]]):
                idx = s.pop()
                res = max(res, i - idx)
        return res
```

### [1081. Smallest Subsequence of Distinct Characters](https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/)

**Example 1:**

```
Input: s = "bcabc"
Output: "abc"
```

```python
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        c = Counter(s)
        visited = set()
        ans = ""
        for ch in s:
            c[ch] -= 1
            if not ans:
                ans += ch
                visited.add(ch)
                continue
            if ch in visited:
                continue
            if ch > ans[-1]:
                ans += ch
                visited.add(ch)
            else:
                while ans and ch < ans[-1] and c[ans[-1]] > 0:
                    tmp = ans[-1]
                    ans = ans[:-1]
                    visited.remove(tmp)
                visited.add(ch)
                ans += ch
        return ans
```


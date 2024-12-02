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

### [1504. Count Submatrices With All Ones](https://leetcode.cn/problems/count-submatrices-with-all-ones/) 矩形

以[i,j]为右下角的全1矩形有多少个呢?

固定列j, row[i]表示左侧连续1的个数

```python
class Solution:
    def numSubmat(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        row = [0] * m
        ans = 0
        for j in range(n):
            # update row
            for i in range(m):
                row[i] = row[i] + 1 if mat[i][j] else 0
 
            # 单调递增栈:
            s = [] # row expand, height
            cumsum = 0 # num of additional matrices with this row as bottom
            for i in range(m):
                currHeight = 1
                while s and row[i] <= s[-1][0]:
                    rowExpand, height = s.pop()
                    cumsum -= (rowExpand - row[i]) * height
                    currHeight += height
                cumsum += row[i] # !
                ans += cumsum
                s.append((row[i], currHeight))
            
        return ans
```

### [1526. Minimum Number of Increments on Subarrays to Form a Target Array](https://leetcode.cn/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/)

**Example 2:**

```
Input: target = [3,1,1,2]
Output: 4
Explanation: [0,0,0,0] -> [1,1,1,1] -> [1,1,1,2] -> [2,1,1,2] -> [3,1,1,2]
```

```python
class Solution:
    def minNumberOperations(self, target: List[int]) -> int:
        s = [0]
        ans = 0
        for h in target:
            if h > s[-1]:
                ans += h - s[-1]
                s.append(h)
            else:
                while h < s[-1]:
                    s.pop()
                s.append(h) 
        return ans
```

### [2030. Smallest K-Length Subsequence With Occurrences of a Letter](https://leetcode.cn/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/)

**Example 2:**

```
Input: s = "leetcode", k = 4, letter = "e", repetition = 2
Output: "ecde"
Explanation: "ecde" is the lexicographically smallest subsequence of length 4 that has the letter "e" appear at least 2 times.
```

```python
class Solution:
    def smallestSubsequence(self, s: str, k: int, letter: str, repetition: int) -> str:
        # pop if (1) ch is smaller (2) have at least k remaining (3) have at least rep letters
        numLetter = 0
        for ch in s:
            if ch == letter:
                numLetter += 1
        stk = ""
        n = len(s)
        idx = None
        needPop = 0
        for i, ch in enumerate(s):
            while stk and ch < stk[-1] and (len(stk) - 1 + n - 1 - i + 1) >= k:
                if stk[-1] == letter:
                    if numLetter - 1 < repetition:
                        break
                    else:
                        numLetter -= 1
                stk = stk[:-1]
            stk += ch
            
            # 超出了，可以pop其他字符，但是不一定可以pop letter，先append再说
            # 先记录在 needPop 里面最后反向 pop 大的

            if len(stk) > k:
                if stk[-1] != letter:
                    stk = stk[:-1]
                else:
                    if numLetter - 1 < repetition:
                        # cannot pop -1, should the max character which != letter
                        k += 1
                        needPop += 1
                    else:
                        # can pop
                        stk = stk[:-1]
                        numLetter -= 1
        addBackLetter = 0
        for i in range(len(stk) - 1, -1, -1):
            if needPop == 0:
                break
            if stk[i] == letter:
                addBackLetter += 1
                stk = stk[:-1]
            else:
                needPop -= 1
                stk = stk[:-1]
        stk += letter * addBackLetter
        return stk
        
```





### [2940. Find Building Where Alice and Bob Can Meet](https://leetcode.cn/problems/find-building-where-alice-and-bob-can-meet/)

If a person is in building `i`, they can move to any other building `j` if and only if `i < j` and `heights[i] < heights[j]`.

You are also given another array `queries` where `queries[i] = [ai, bi]`. On the `ith` query, Alice is in building `ai` while Bob is in building `bi`.

Return *an array* `ans` *where* `ans[i]` *is **the index of the leftmost building** where Alice and Bob can meet on the* `ith` *query*. *If Alice and Bob cannot move to a common building on query* `i`, *set* `ans[i]` *to* `-1`.

 

```python
class Solution:
    def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
        # j之后 第一次大于 max(q1,q2) 的位置
        # 如果q1>q2
        # q1 <= q2: 答案就是j
        n = len(queries)
        ans = [-1] * n
        sortedQ = []
        for i, (idx1, idx2) in enumerate(queries):
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            if idx1 == idx2:
                ans[i] = idx1
            elif heights[idx1] < heights[idx2]:
                ans[i] = idx2
            else:
                sortedQ.append((idx2, idx1, i))
        sortedQ.sort(reverse=True)
        s = []
    
        def findLeftMostGE(s, x):
            l, r = 0, len(s) - 1
            while l < r:
                mid = (l + r) // 2
                midValue = heights[s[mid]]
                if midValue == x:
                    l, r = mid, mid
                elif midValue > x:
                    l, r = mid + 1, r
                else:
                    l, r = l, mid - 1
            if l > r:
                l, r = r, l
            l = max(l-1, 0)
            r = min(r+1, len(s) - 1)
            for idx in range(r, l-1,-1):
                if heights[s[idx]] > x:
                    return s[idx]
            return -1
            
        j = len(heights) - 1
        for idx2, idx1, i in sortedQ:
            while j >= idx2:
                while s and heights[j] >= heights[s[-1]]:
                    s.pop()
                s.append(j)
                j -= 1
            pos = findLeftMostGE(s, heights[idx1])
            ans[i] = pos
        return ans

```

### [2945. Find Maximum Non-decreasing Array Length](https://leetcode.cn/problems/find-maximum-non-decreasing-array-length/)

可以合并子数组，nums[j+1...i] 变成他们的和。要求单调递增。问最长有多少个元素

```python
class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # dp[i] >= dp[i-1] >... 因为总是可以把最后一个元素加入最后一组，至少保持相同的长度
        
        # [0,...,j] | [j+1,...,i]
        # s[i] - s[j] >= last[j]
        # 找到一个j
        # such that s[i] >= last[j] + s[j] 并且 j 尽量大
        # 由于s[i] 单调递增，所以那些不满足条件的j可以删去
        # 如果 last[j] + s[j] <= last[j-1] + s[j-1] 那么删去j-1
        # j = 0,... n 在stack中，那么last[j] + s[j] 必须严格单调递增,/

        # 来了一个 s[i] , 找到最后边的 j, such that  s[i] >= last[j] + s[j]
        # j 之前的那些都可删去，因为下一个s[i]更大，之前的last[j] + s[i] 都太小了, j也太小了
        # 找到最后的j: 从头开始删去, as long as s[i] >= last[j] + s[j] 并且 删去后还有一组

        # s[-1]=0, last[-1]=0

        
        n = len(nums)
        s = [0] * n
        s[0] = nums[0]
        for i in range(1, n):
            s[i] = s[i - 1] + nums[i]
        
        dp = [0] * n
        q = deque([(-1, 0)]) # 存 (j, last[j]+s[j])
        for i in range(n):
            while len(q) >= 2 and q[1][1] <= s[i]: # pop 之后还有满足条件的
                q.popleft()
            j, _ = q[0] # 只留下满足条件的最后一个
            currLast = s[i] - (s[j] if j != -1 else 0)
            currSum = currLast + s[i]
            while q and currSum <= q[-1][1]: # 删去次优的j
                q.pop()
            q.append((i, currSum))
            dp[i] = (dp[j] if j != -1 else 0) + 1

        return dp[n - 1]
```

### [3205. Maximum Array Hopping Score I](https://leetcode.cn/problems/maximum-array-hopping-score-i/)

In each **hop**, you can jump from index `i` to an index `j > i`, and you get a **score** of `(j - i) * nums[j]`.

```python
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # 1 2 1 4 [5] 1 3 2 [4] 1 [2] 1 [0]
        n = len(nums)
        s = [n - 1]
        for i in range(n - 2, -1, -1):
            if nums[i] > nums[s[-1]]:
                s.append(i)
        # s is the jump targets
        if s[-1] != 0:
            s.append(0)
        ans = 0
        for i in range(len(s) - 1):
            ed, st = s[i], s[i + 1]
            ans += (ed - st) * nums[ed]
        return ans
```

### [3229. Minimum Operations to Make Array Equal to Target](https://leetcode.cn/problems/minimum-operations-to-make-array-equal-to-target/) 差分数组

x = targets - nums

构造x from [0...0]

diff元素+1, 那么可以对后面的免费-1

diff元素-1, 那么可以对后面的免费+1

记diff为`x[0], x[1]-x[0] .... ,0-x[n-1]` 总和为0。最后一个元素一般不需要。

考虑diff数组中正的负的，如果第一个数x>0，那么一定给后面负数免费贡献了-x. 正负总是相抵。

只需要返回diff中正数和。

```python
class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        targets = [0] + [t - num for num, t in zip(nums, target)] + [0]
        diff = [y - x for x, y in pairwise(targets)]
        # diff 总和是 0
        return sum([x for x in diff if x > 0])  
```

### [面试题 03.05. 栈排序](https://leetcode.cn/problems/sort-of-stacks-lcci/)

维护stack最小数字在顶。能不倒就不倒。除非peek/pop，或者加入新元素时，新元素太小。

```python
class SortedStack:

    def __init__(self):
        self.s1 = []
        self.s2 = []
        # [9, 5, 3, 1] 加入 4
        # [9, 5, 4]   [1 3]
 
    def push(self, val: int) -> None:
        s1, s2 = self.s1, self.s2
        while s2 and val < s2[-1]:
            s1.append(s2.pop())
        while s1 and val > s1[-1]:
            s2.append(s1.pop())
        s1.append(val)

    def pop(self) -> None:
        s1, s2 = self.s1, self.s2
        while s2:
            s1.append(s2.pop())
        if s1:
            s1.pop()

    def peek(self) -> int:
        s1, s2 = self.s1, self.s2
        while s2:
            s1.append(s2.pop())
        return s1[-1] if s1 else -1
        

    def isEmpty(self) -> bool:
        s1, s2 = self.s1, self.s2
        return len(s1) == 0 and len(s2) == 0
        


# Your SortedStack object will be instantiated and called as such:
# obj = SortedStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.peek()
# param_4 = obj.isEmpty()
```


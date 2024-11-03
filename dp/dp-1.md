## 动态规划：线性&区间动态

[TOC]

### [300. Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 动态规划 + 二分查找
        # dp 保存长度为 i 的上升子序列末位数的最小值
        # dp 是单调递增的
        dp = [-float('inf')]

        def bs(x):
            l, r = 0, len(dp)
            while l < r:
                mid = (l + r) // 2
                if dp[mid] == x:
                    break
                elif dp[mid] > x:
                    r = mid - 1
                else:
                    l = mid + 1
            if dp[(l + r) // 2] == x:
                return (l + r) // 2
            if dp[l] < x:
                l += 1
            return l
        
        for x in nums:
            if x > dp[-1]:
                dp.append(x)
            else:
                idx = bs(x)
                dp[idx] = x
        return len(dp) - 1

        # # 动态规划 N^2
        # dp = [1] * len(nums)
        # for i in range(len(nums)):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        # return max(dp)
```



### [673. Number of Longest Increasing Subsequence](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/)

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[-float('inf')]]
        cnt = [[1]]

        def bs1(x, arr):
            l, r = 0, len(arr) - 1
            while l < r:
                mid = (l + r) // 2
                if arr[mid] == x:
                    break
                elif arr[mid] > x:
                    r = mid - 1
                else:
                    l = mid + 1
            if arr[(l + r) // 2] == x:
                return (l + r) // 2 - 1
            if arr[l] > x:
                l -= 1
            return l

        def bs2(x, arr):
            l, r = 0, len(arr) - 1
            while l < r:
                mid = (l + r) // 2
                if arr[mid] == x:
                    break
                elif arr[mid] > x:
                    l = mid + 1
                else:
                    r = mid - 1
            if arr[(l + r) // 2] == x:
                return (l + r) // 2 + 1
            if arr[l] > x:
                l += 1
            return l

        for x in nums:
            # 找到一个最后一个元素严格比他小的row
            idx1 = bs1(x, [i[-1] for i in dp])
            # 这个row从哪里开始严格比他小
            idx2 = bs2(x, dp[idx1])
            add_cnt = cnt[idx1][-1] - cnt[idx1][idx2 - 1] if idx2 - 1 >= 0 else cnt[idx1][-1] # add_cnt = sum(cnt[idx1][idx2:]) 
            # 更新到 idx1 + 1行
            if idx1 == len(dp) - 1:
                cnt.append([add_cnt])
                dp.append([x])
            else:
                if x == dp[idx1 + 1][-1]:
                    cnt[idx1 + 1][-1] += add_cnt
                else:
                    dp[idx1 + 1].append(x)
                    cnt[idx1 + 1].append(add_cnt + cnt[idx1 + 1][-1]) # cnt[idx1 + 1].append(add_cnt)
        
        return cnt[-1][-1] # sum(cnt[-1])
            
 

        # N^2 算法
        # n = len(nums)
        # dp = [1] * n  # 长度
        # cnt = [1] * n # 个数
        # best_len = 1
        # best_cnt = 1
        # for i in range(n):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             if dp[j] + 1 > dp[i]:
        #                 dp[i] = dp[j] + 1
        #                 cnt[i] = cnt[j]
        #             elif dp[j] + 1 == dp[i]:
        #                 cnt[i] += cnt[j]
        #     if dp[i] > best_len:
        #         best_len, best_cnt = dp[i], cnt[i]
        # return best_cnt
```



### [354. Russian Doll Envelopes](https://leetcode.cn/problems/russian-doll-envelopes/)

One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

1. sorted by (width, -height) 确保相同 width，height 递减，从而不会出现一边相同，一边严格上升的情况
2. LIS of height

### [152. Maximum Product Subarray](https://leetcode.cn/problems/maximum-product-subarray/)

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        prev_max = nums[0]
        prev_min = nums[0]
        ans = nums[0]
        for x in nums[1:]:
            arr = [x, x * prev_max, x * prev_min]
            prev_max, prev_min = max(arr), min(arr)
            ans = max(ans, prev_max)
        return ans
```



### [918. Maximum Sum Circular Subarray](https://leetcode.cn/problems/maximum-sum-circular-subarray/)

```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)

        # 最大和
        max_s = nums[0]
        prev = nums[0]
        for x in nums[1:]:
            prev = max(prev + x, x)
            max_s = max(max_s, prev)
        
        # 总的扣去最小和 1 xxxxx n
        min_s = 0
        prev = 0
        for x in nums[1: -1]:
            prev = min(prev + x, x)
            min_s = min(min_s, prev)
     
        return max(max_s, sum(nums) - min_s)
```

### [面试题 17.24. Max Submatrix LCCI](https://leetcode.cn/problems/max-submatrix-lcci/)

Find the submatrix with the largest possible sum.

```python
import numpy as np
class Solution:
    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])

        # 遍历 row i->j 的组合，求 column sum 的最大子数组
        
        for i in range(1, m):
            for j in range(n):
                matrix[i][j] += matrix[i - 1][j]

        def colSum(i, j):
            if i - 1 >= 0:
                return [matrix[j][k] - matrix[i - 1][k] for k in range(len(matrix[j]))]
            return matrix[j]

        def maxSubArr(arr):
            prevSum = arr[0]
            prevStart = 0
            bestSum = arr[0]
            bestStart = 0
            bestEnd = 0
            for i in range(1, len(arr)):
                x = arr[i]
                if prevSum > 0:
                    prevSum = prevSum + x
                else:
                    prevSum = x
                    prevStart = i
                if prevSum > bestSum:
                    bestSum = prevSum
                    bestStart = prevStart
                    bestEnd = i
            return bestStart, bestEnd, bestSum

        r1, r2 = 0, 0
        c1, c2 = 0, 0
        bestSum = matrix[0][0]

        for i in range(m):
            for j in range(i, m):
                arr = colSum(i, j)
                st, ed, curr = maxSubArr(arr)
                if curr > bestSum:
                    bestSum = curr
                    r1, r2 = i, j
                    c1, c2 = st, ed

        return [r1, c1, r2, c2]

```



### [363. Max Sum of Rectangle No Larger Than K](https://leetcode.cn/problems/max-sum-of-rectangle-no-larger-than-k/)

枚举矩形的上下边界，并计算出该边界内每列的元素和，则原问题转换成了如下一维问题：

给定一个整数数组和一个整数 *k*，计算该数组的最大区间和，要求区间和不超过 *k*。

$S_r - S_l \le k$ 

=> $S_l \ge S_r - k$  尽可能小的 $S_l$



```c++
class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>> &matrix, int k) {
        int ans = INT_MIN;
        int m = matrix.size(), n = matrix[0].size();
        for (int i = 0; i < m; ++i) { // 枚举上边界
            vector<int> sum(n);
            for (int j = i; j < m; ++j) { // 枚举下边界
                for (int c = 0; c < n; ++c) {
                    sum[c] += matrix[j][c]; // 更新每列的元素和
                }
                set<int> sumSet{0};
                int s = 0;
                for (int v : sum) {
                    s += v;
                    auto lb = sumSet.lower_bound(s - k);
                    if (lb != sumSet.end()) {
                        ans = max(ans, s - *lb);
                    }
                    sumSet.insert(s);
                }
            }
        }
        return ans;
    }
};



from sortedcontainers import SortedList

class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        ans = float("-inf")
        m, n = len(matrix), len(matrix[0])

        for i in range(m):   # 枚举上边界
            total = [0] * n
            for j in range(i, m):   # 枚举下边界
                for c in range(n):
                    total[c] += matrix[j][c]   # 更新每列的元素和
                
                totalSet = SortedList([0])
                s = 0
                for v in total:
                    s += v
                    lb = totalSet.bisect_left(s - k)
                    if lb != len(totalSet):
                        ans = max(ans, s - totalSet[lb])
                    totalSet.add(s)

        return ans
```

 


# 动态规划：线性&区间动态

[TOC]

## 单串

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

 

### [1388. Pizza With 3n Slices](https://leetcode.cn/problems/pizza-with-3n-slices/)

```python
class Solution:
    def maxSizeSlices(self, slices: List[int]) -> int:
        # 给一个长度为 3n 的环状序列，选 n 个数，任意两个数不能相邻，求这 n 个数和的最大值。

        # f[i][j] 表示在长度为i的数组中选取j个不相邻的点所能取得的最大收益
        # j <= (i + 1) // 2
        # f[i][j] = gain[i-1] + f[i-2][j-1] or f[i-1][j]
        
        def run(arr, k):
            f = [[0]]
            for i in range(1, len(arr) + 1):
                f.append([])
                for j in range(0, (i + 1) // 2 + 1):
                    if j == 0:
                        f[-1].append(0)
                        continue
                    tmp = float('-inf')
                    if j < len(f[i - 1]):
                        tmp = max(tmp, f[i - 1][j])
                    v1 = arr[i - 1]
                    v2 = 0 if j == 1 else f[i - 2][j - 1] if i - 2 >= 0 and j - 1 < len(f[i - 2]) else float('-inf')
                    tmp = max(tmp, v1 + v2)
                    f[-1].append(tmp)
            return f[-1][k]
        
        return max(run(slices[1:], len(slices) // 3), run(slices[:-1], len(slices) // 3)) # 第一个和最后一个不能同时取到
```

### [873. Length of Longest Fibonacci Subsequence](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/)

```
Input: arr = [1,2,3,4,5,6,7,8]
Output: 5
Explanation: The longest subsequence that is fibonacci-like: [1,2,3,5,8].
```

```python
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        f = defaultdict(int)
        max_len = 0
        lastAppear = {}
        for i, x in enumerate(arr):
            lastAppear[x] = i
    
        for i in range(2, len(arr)):
            for j in range(1, i):
                ask = arr[i] - arr[j]
                idx = lastAppear.get(ask, -1)
                if idx >= j or idx == -1:
                    continue
                # idx -> j -> i 是一组，判断 .... -> idx -> j能有多少组
                f[(j, i)] = max(f[(j, i)], f.get((idx, j), 2) + 1)
                max_len = max(max_len, f[(j, i)])
        return max_len  
```



### [1027. 最长等差数列](https://leetcode.cn/problems/longest-arithmetic-subsequence/)

```python
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        diff = max(nums) - min(nums)
        ans = 1
        for d in range(-diff, diff + 1): # 枚举等差 d
            f = dict()
            for x in nums:
                # x 单独成为一个序列 /
                # x -> (x-d)
                tmp = 1 + f[x - d] if x - d in f else 1
                f[x] = max(tmp, f.get(x, 1))
                ans = max(ans, tmp)
        return ans
```



### [1055. 形成字符串的最短路径](https://leetcode.cn/problems/shortest-way-to-form-string/)

```
输入：source = "xyz", target = "xzyxz"
输出：3
解释：目标字符串可以按如下方式构建： "xz" + "y" + "xz"。
```

```python
import bisect

class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        src_set = set(source)
        for x in target:
            if x not in src_set:
                return -1

        srcPlaces = defaultdict(list)
        for i, x in enumerate(source):
            srcPlaces[x].append(i)
        
        ans = 1
        srcIdx = 0
        for t in target:
            # [....] 中 >= srcIdx 的第一个数
            i = bisect.bisect_left(srcPlaces[t], srcIdx)
            if i == len(srcPlaces[t]):
                # 需要重新开启一个新串
                ans += 1
                srcIdx = bisect.bisect_left(srcPlaces[t], 0)
                srcIdx = srcPlaces[t][srcIdx]
            else:
                srcIdx = srcPlaces[t][i]
            srcIdx += 1
        return ans
```



### [368. 最大整除子集](https://leetcode.cn/problems/largest-divisible-subset/)

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums = sorted(nums)
        f = [1] * len(nums)
        prev = [i for i in range(len(nums))]
        bestL = 1
        bestIdx = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if f[j] + 1 > f[i]:
                        f[i] = f[j] + 1
                        prev[i] = j
                        if f[i] > bestL:
                            bestL = f[i]
                            bestIdx = i
        res = []
        while True:
            res.append(nums[bestIdx])
            if bestIdx == prev[bestIdx]:
                break
            bestIdx = prev[bestIdx]
        reversed(res)
        return res
```

### [32. Longest Valid Parentheses](https://leetcode.cn/problems/longest-valid-parentheses/)

Given a string containing just the characters `'('` and `')'`, return *the length of the longest valid (well-formed) parentheses* *substring*.

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        leftCnt, rightCnt = 0, 0
        res = 0
        for ch in s:
            if ch == '(':
                leftCnt += 1
            else:
                rightCnt += 1
            if leftCnt == rightCnt:
                res = max(res, leftCnt)
            elif rightCnt > leftCnt:
                leftCnt, rightCnt = 0, 0
 
        leftCnt, rightCnt = 0, 0
        for ch in s[::-1]:
            if ch == '(':
                leftCnt += 1
            else:
                rightCnt += 1
            if leftCnt == rightCnt:
                res = max(res, leftCnt)
            elif leftCnt > rightCnt:
                leftCnt, rightCnt = 0, 0

        return res * 2
    
    
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stk = [(-1, ')')]
        res = 0
        for i, ch in enumerate(s):
            if ch == '(':
                stk.append((i, ch))
            else:
                _, head = stk.pop()
                if head == '(':
                    res = max(res, i - stk[-1][0])
                    print(i, stk[-1][0])
                else:
                    stk.append((i, ')'))
        return res
```



[338. Counting Bits](https://leetcode.cn/problems/counting-bits/)

Given an integer `n`, return *an array* `ans` *of length* `n + 1` *such that for each* `i` (`0 <= i <= n`)*,* `ans[i]` *is the **number of*** `1`***'s** in the binary representation of* `i`.

**Example 2:**

```
Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 -->   0
1 -->   1
2 -->  10
3 -->  11
4 --> 100
5 --> 101
```

```python
class Solution:
    def countBits(self, n: int) -> List[int]:
        # x:  ...... 1 0 0
        # x-1:.......0 1 1

        # x & (x-1) set the lowest 1 in x to 0
        # f[x] = f[x & (x-1)] + 1
        f = [0] * (n + 1)
        for i in range(1, n + 1):
            f[i] = f[i & (i - 1)] + 1
        return f
```

### [871. 最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/)

```python
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:

        # # f[i]: 加油 i 次能走多远. O(N2)
        # dp = [0] * (len(stations) + 1)
        # dp[0] = startFuel
        # if dp[0] >= target:
        #     return 0
        
        # for i, [stationPos, fuel] in enumerate(stations): # 枚举最后一次加油的位置
        #     # 第 i 个加油站前最多加油 i 次
        #     for j in range(i, -1, -1):
        #         if dp[j] >= stationPos:
        #             dp[j + 1] = max(dp[j + 1], dp[j] + fuel)
        # return next((i for i, v in enumerate(dp) if v >= target), -1)

        # 贪心
        fuelRemain = startFuel
        prevPos = 0
        ans = 0
        fuelCandidates = []
        stations.append((target, 0))
        
        for i, (stationPos, fuel) in enumerate(stations):
            fuelRemain -= (stationPos - prevPos)
            while fuelRemain < 0 and fuelCandidates:
                selectedFuel, selectedPos = heapq.heappop(fuelCandidates)
                selectedFuel = -selectedFuel
                fuelRemain += selectedFuel
                ans += 1
            if fuelRemain < 0:
                return -1
            heapq.heappush(fuelCandidates, (-fuel, stationPos))
            prevPos = stationPos
        return ans
```

### [813. Largest Sum of Averages](https://leetcode.cn/problems/largest-sum-of-averages/)

You are given an integer array `nums` and an integer `k`. You can partition the array into **at most** `k` non-empty adjacent subarrays. The **score** of a partition is the sum of the averages of each subarray.

Return *the maximum **score** you can achieve of all the possible partitions*. 

首先证明需要用尽 k 因为都是正数。

```python
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        # dp[i][j]: 枚举最后一组的起始下标 lastStart, lastStart之前一共有lastStart个数字, lastStart >= j - 1
        # 前 i 个数字 分成 j 组
        # j = 1, 2, ..., i
        s = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            s[i + 1] = s[i] + x
        def avg(i, j):
          
            return (s[j + 1] - s[i]) / (j - i + 1)
        
        dp = [[0] * (k + 1) for i in range(len(nums) + 1)]
        for i in range(1, len(nums) + 1):
            for j in range(1, min(i + 1, k + 1)):
                dp[i][j] = 0
                for lastStart in range(j - 1, i):
                    if lastStart > 0 and j - 1 == 0: continue
                    dp[i][j] = max(dp[i][j], dp[lastStart][j - 1] + avg(lastStart, i - 1))
        return dp[len(nums)][k]

```

### [887. Super Egg Drop](https://leetcode.cn/problems/super-egg-drop/)

<img src="images\egg-floor.PNG" alt="egg-floor" style="zoom:51%;" />

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        # k * n * log(n)
        d = {}
        def f(k, n):
            if (k, n) in d:
                return d[(k, n)]
            if n == 0:
                return 0
            if k == 1:
                return n
            l, r = 1, n
            
            while l < r:
                mid = (l + r) // 2
                v1 = f(k - 1, mid - 1)
                v2 = f(k, n - mid)
                if v1 == v2:
                    d[(k, n)] = 1 + v1
                    return d[(k, n)] 
                elif v1 < v2:
                    l, r = mid + 1, r
                else:
                    l, r = l, mid - 1
            # 现在 l == r
            if l - 1 > 0:
                l -= 1

            d[(k, n)] = 1 + min(max(f(k - 1, l - 1), f(k, n - l)), max(f(k - 1, r - 1), f(k, n - r)))
            
            return d[(k, n)]
        
        return f(k, n)

        # (k * n)
        # 固定k, x_opt随着n单调增加
        f1 = [i for i in range(n + 1)] # k=1 只有一个鸡蛋，最差情况下从下而上走完所有层数
        f2 = [i for i in range(n + 1)]
        for egg in range(2, min(k, n) + 1):
            ans = 1
            for floor in range(2, n + 1):
                for x in range(ans, floor + 1):
                    if f1[x - 1] > f2[floor - x]:
                        break
                l, r = x - 1, x
                if max(f1[l - 1], f2[floor - l]) < max(f1[r - 1], f2[floor - r]):
                    f2[floor] = max(f1[l - 1], f2[floor - l]) + 1
                    ans = l
                else:
                    f2[floor] = max(f1[r - 1], f2[floor - r]) + 1
                    ans = r
            f1 = f2[:]
        return f1[-1]


        # 方法3:
        # f[t次][k个鸡蛋] 最多能区分多少层
        # 不必思考到底在哪里扔这个鸡蛋，我们只需要扔出一个鸡蛋，看看到底发生了什么
        # 如果鸡蛋没有碎，在上方能区分 [t-1][k]
        # 如果鸡蛋碎了，在下方能区分 f[t-1][k-1]
        # f[t][k] = 1 + f[t-1][k-1] + f[t-1][k]
        
        if n == 1:
            return 1
        
        f = [1] * (k + 1) # t = 1
        f[0] = 0
        for t in range(2, n + 1):
            g = [0] * (k + 1)
            for j in range(1, k + 1):
                g[j] = 1 + f[j - 1] + f[j]
            if g[k] >= n:
                return t
            f = g[:]
```

### [975. Odd Even Jump](https://leetcode.cn/problems/odd-even-jump/) 单调栈求next数组

```python
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        oddNext = [-1] * len(arr) # 最小的（然后是最近的）>= 自己的数字
        evenNext = [-1] * len(arr)

        # 数字从小到大排，相同则小index优先
        N = len(arr)
        inc = sorted([i for i in range(N)], key=lambda i: (arr[i], i))
        s = []
        for idx in inc:
            while s:
                loc = s[-1]
                if loc < idx:
                    oddNext[loc] = idx
                    s.pop()
                else:
                    break
            s.append(idx)
        dec = sorted([i for i in range(N)], key=lambda i: (-arr[i], i))
        s = []
        for idx in dec:
            while s:
                loc = s[-1]
                if loc < idx:
                    evenNext[loc] = idx
                    s.pop()
                else:
                    break
            s.append(idx)

        oddReachTail = [False] * (len(arr) - 1) + [True]
        evenReachTail = [False] * (len(arr) - 1) + [True]
        for idx in range(len(arr) - 2, -1, -1):
            if oddNext[idx] != -1:
                oddReachTail[idx] = evenReachTail[oddNext[idx]]
            if evenNext[idx] != -1:
                evenReachTail[idx] = oddReachTail[evenNext[idx]]
        return sum(oddReachTail)
```

### [1478. 安排邮筒](https://leetcode.cn/problems/allocate-mailboxes/)

```python
class Solution:
    def minDistance(self, houses: List[int], k: int) -> int:
        # dp[i][j]: 前i个房子分配j个筒 (1<=j<=i)
        # dp[i][i] = 0
        # dp[i][j] = min_{最后一组有x个元素} dp[i-x][j-1] + cost(i-x+1, i)

        houses =sorted(houses)

        N = len(houses)
        if k >= N:
            return 0
    
        s = [0]
        for dist in houses:
            s.append(dist + s[-1])
    
        def cost(i, j):
            if i == j:
                return 0
            mid = (j + i) // 2
            if (j - i + 1) % 2:
                return s[j] - s[mid] - s[mid - 1] + s[i - 1]
            else:
                return s[j] - s[mid] - s[mid] + s[i - 1]

        dp = {(0, 0): 0}
        for i in range(1, N + 1):
            for j in range(1, min(i, k) + 1):
                for x in range(1, i - j + 2): # 前面没有冗余的筒，前面的筒有j-1个，后面最多有 i-(j-1)=i-j+1个元素
                    dp[(i, j)] = min(dp.get((i, j), float('inf')), dp.get((i - x, j - 1), float('inf')) + cost(i - x + 1, i))
        return dp[(N, k)]
```



### [410. 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)

```python
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        
        nums = [x for x in nums if x > 0]
        k = min(k, len(nums))
        if len(nums) == 0:
            return 0
        
        # f[前i个数][分成j份] = min_x:: max{f[前x个数][分成j-1份], sum(第x+1个数 -> 第i个数)}
        # f = {(0, 0): 0}
        # N = len(nums)
        # s = [0]
        # for num in nums:
        #     s.append(s[-1] + num)
        
        # for i in range(1, N + 1):
        #     for j in range(1, min(k, i) + 1):
        #         f[(i, j)] = float('inf')
        #         for x in range(j - 1, i):
        #             f[(i, j)] = min(
        #                 f[(i, j)],
        #                 max(f.get((x, j - 1), float('inf')), s[i] - s[x])
        #                 )
        # return f[(N, k)]

        # 二分: max(nums) -> sum(nums)
        def check(maxSum, k):
            cnt = 1
            tmp = 0
            for i, num in enumerate(nums):
                if num > maxSum:
                    return False
                tmp += num
                if tmp > maxSum:
                    cnt += 1
                    tmp = num
                    if cnt > k:
                        return False
            return True

        l = max(nums)
        r = sum(nums)
        ans = float('inf')
        while l <= r:
            mid = (l + r) // 2
            if check(mid, k):
                ans = min(ans, mid)
                r = mid - 1
            else:
                l = mid + 1
        return ans
```

### [1473. Paint House III](https://leetcode.cn/problems/paint-house-iii/)

把房子正好涂成target个block，每个block的颜色相同。有的房子已经有颜色了。

```python
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        # f[i个house][j个街区][最后一个颜色是??]
        # f[i-1][j][最后一个颜色是k] & 第i个house也要涂成k
        # f[i-1][j-1][最后一个颜色是k] & 第i个house涂其他颜色

        # first house, paintable colors
        f = {}
        def availableColors(i):
            if houses[i - 1]:
                return (houses[i - 1],)
            return range(1, n + 1)

        if houses[0]:
            f[(1, houses[0])] = 0
        else:
            for color in range(1, n + 1):
                f[(1, color)] = cost[0][color - 1]

        for i in range(2, len(houses) + 1):
            g = {}
            for j in range(1, min(target, i) + 1):
                for color in availableColors(i):
                    addCost = 0 if houses[i - 1] else cost[i - 1][color - 1] 
                    # 上一个同色
                    if (j, color) in f:
                        g[(j, color)] = f[(j, color)] + addCost
                    # 上一个不同色:
                    for diffColor in range(1, n + 1):
                        if diffColor != color and (j - 1, diffColor) in f:
                            g[(j, color)] = min(
                                g.get((j, color), float('inf')),
                                f[(j - 1, diffColor)] + addCost
                            )
            if not g:
                return -1
            f = g
        ans = float('inf')
        for color in availableColors(len(houses)):
            ans = min(ans, f.get((target, color), float('inf')))
        return ans if ans != float('inf') else -1
```

### [188. Best Time to Buy and Sell Stock IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)

Find the maximum profit you can achieve. You may complete at most `k` transactions: i.e. you may buy at most `k` times and sell at most `k` times.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

```python
import numpy as np

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        k = min(k , len(prices) // 2)
        if k <= 0: return 0
        buy = [float('-inf')] * (k + 1)
        sell = [float('-inf')] * (k + 1)
        buy[1] = -prices[0]
        sell[0] = 0
        for idx, p in enumerate(prices[1:]):
            for i in range(1, min(k, (idx+ 2)//2 + 1) + 1):
                buy[i], sell[i] = max(buy[i], sell[i - 1] - p), max(sell[i], buy[i] + p)
        return max(sell)
```

### [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

任意多transactions

After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = -prices[0]
        sell = 0
        gap = 0
        for price in prices[1:]:
            # buy -> 维持原状 buy
            # buy -> sell 获得 price
            # sell -> gap
            # gap -> buy, pay price
            # gap -> gap
            buy, sell, gap = max(buy, gap - price), buy + price, max(sell, gap)
        return max(sell, gap)
```

### [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        buy, sell = -prices[0] - fee, 0
        for price in prices[1:]:
            buy, sell = max(buy, sell - price - fee), max(sell, buy + price)
        return sell
```

## 双串

### [1143. Longest Common Subsequence](https://leetcode.cn/problems/longest-common-subsequence/)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp[i个字符][j个字符]
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp[i个字符][j个字符]
        m, n = len(text1), len(text2)
        f = [0] * (n + 1)
        for i in range(1, m + 1):
            g = [0] * (n + 1)
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    g[j] = 1 + f[j - 1]
                else:
                    g[j] = max(f[j], g[j - 1])
            f = g
        return g[n]
```

### [10. Regular Expression Matching](https://leetcode.cn/problems/regular-expression-matching/)

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m = len(s)
        n = len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[m][n] = True

        # p = a*.*.*
        for j in range(n - 1, -1, -1):
            dp[m][j] = j + 1 < n and p[j + 1] == '*' and dp[m][j + 2]
        
        def matchRegChar(x, y ):
            return x == y or x == '.' or y == '.'
  
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if p[j] == '*':
                    continue
                if j + 1 >= n or p[j + 1] != '*': 
                    dp[i][j] = dp[i + 1][j + 1] and matchRegChar(s[i], p[j])
                else:
                    # 下一个是 *
                    # use once, reuse, not use
                    if matchRegChar(s[i], p[j]):
                        dp[i][j] = dp[i + 1][j + 2] or dp[i + 1][j]
                    dp[i][j] = dp[i][j] or dp[i][j + 2]
        return dp[0][0]
```

### [97. Interleaving String](https://leetcode.cn/problems/interleaving-string/)

s1 s2是否能交错组成s3

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        # [已经匹配的] (最后一个元素 要么来自s1, 要么来自s2)
        # f[s1的前i个元素][f2的前j个元素] 是否能match s3 的前i+j-1个元素
        # = f[i-1][j] & s1[i-1] == s3[i+j-1] 
        #   or f[i][j-1] & s2[j-1] == s3[i+j-1]
        f = [False] * (len(s2) + 1)
        f[0] = True
        for idx, ch in enumerate(s2):
            f[idx + 1] = f[idx] and idx < len(s3) and s2[idx] == s3[idx]
        
        for i in range(1, len(s1) + 1):
            g = [False] * (len(s2) + 1)
            g[0] = f[0] and i - 1 < len(s3) and s1[i - 1] == s3[i - 1]
            for j in range(1, len(s2) + 1):
                g[j] = f[j] and i + j - 1 < len(s3) and s1[i - 1] == s3[i + j - 1]
                g[j] = g[j] or g[j - 1] and i + j - 1 < len(s3) and s2[j - 1] == s3[i + j - 1]
            f = g
        return f[len(s2)]
```

### [87. Scramble String](https://leetcode.cn/problems/scramble-string/)

匹配两个字符串：找一个位置切开，切开后可以左右互换

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        @cache
        def dfs(idx1, idx2, length):
            if s1[idx1: idx1 + length] == s2[idx2: idx2 + length]:
                return True
            if Counter(s1[idx1: idx1 + length]) != Counter(s2[idx2: idx2 + length]):
                return False
            for leftLen in range(1, length):
                rightLen = length - leftLen
                # no swap
                if dfs(idx1, idx2, leftLen) and dfs(idx1 + leftLen, idx2 + leftLen, length - leftLen):
                    return True
                # swap
                if dfs(idx1, idx2 + length - leftLen, leftLen) and dfs(idx1 + leftLen, idx2, rightLen):
                    return True
            return False
        ans = dfs(0, 0, len(s1))
        dfs.cache_clear()
        return ans
```

## 矩阵

### [174. Dungeon Game](https://leetcode.cn/problems/dungeon-game/)

The knight has an initial health point represented by a positive integer. If at any point his health point drops to `0` or below, he dies immediately. Return *the knight's minimum initial health so that he can rescue the princess*.

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])
        # f[i][j]表示从这个点出发达到终点，至少需要多少initial health
        f = [float('inf')] * n
        f[-1] = 1
        for i in range(m - 1, -1, -1):
            g = [0] * n
            for j in range(n - 1, -1, -1):
                g[j] = max(min(f[j], g[j + 1] if j + 1 < n else float('inf')) - dungeon[i][j], 1)
            f = g
        return f[0]
```


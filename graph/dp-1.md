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


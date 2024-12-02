# 单调栈 - 序

[TOC]

### [42. Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)

<img src="images/rain.PNG" alt="rain water" style="zoom:33%;" />

```python
# 记录左右（包括自己的）最大值
class Solution:
    def trap(self, height: List[int]) -> int:
        leftMax = [height[0]]
        rightMax = [height[-1]]
        for i in range(1, len(height)):
            leftMax.append(max(leftMax[-1], height[i]))
        for i in range(len(height) - 2, -1, -1):
            rightMax.append(max(rightMax[-1], height[i]))
        rightMax.reverse()
        res = 0
        for i in range(len(height)):
            res += min(leftMax[i], rightMax[i]) - height[i]
        return res

# 单调递减栈，如果来了个大的，则形成雨水，pop
class Solution:
    def trap(self, height: List[int]) -> int:
        s = [0]
        res = 0
        for i in range(1, len(height)):
            while len(s) and height[s[-1]] <= height[i]:
                topIdx = s.pop()
                if len(s):
                    res += (min(height[s[-1]], height[i]) - height[topIdx]) * (i - s[-1] - 1)
            s.append(i)
        return res

# 双指针，更新包括自己的leftMax, rightMax，移动小者
class Solution:
    def trap(self, height: List[int]) -> int:
        i, j = 0, len(height) - 1
        leftMax, rightMax = 0, 0
        res = 0
        while i <= j:
            leftMax = max(leftMax, height[i])
            rightMax = max(rightMax, height[j])
            if leftMax >= rightMax:
                res += rightMax - height[j]
                j -= 1
            else:
                res += leftMax - height[i]
                i += 1
        return res
```



### [84. Largest Rectangle in Histogram](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

<img src="images\max-rectangle-area.PNG" alt="max-rectangle-area.PNG" style="zoom:33%;" />

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 想知道，左边第一个比我小的数，右边第一个比我小的数
        n = len(heights)
        leftIdx = [-1] * n
        rightIdx = [n] * n

        s = []
        for i in range(n):
            while s and heights[s[-1]] >= heights[i]:
                s.pop()
            leftIdx[i] = s[-1] if s else -1
            s.append(i)

        s = []
        for i in range(n - 1, -1, -1):
            while s and heights[s[-1]] >= heights[i]:
                s.pop()
            rightIdx[i] = s[-1] if s else n
            s.append(i)
        
        res =  0
        for i in range(n):
            tmp = (rightIdx[i] - leftIdx[i] - 1) * heights[i]
            res = max(res, tmp)
        return res
 
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 想知道，左边第一个比我小的数，右边第一个比我小的数
        n = len(heights)
        leftIdx = [-1] * n # left nearest < me
        rightIdx = [n] * n # right nearest <= me (考虑 xxxx, 只有最后一个柱子正确求出了右边界. -1, n)
        s = [] # mono increase
        for i in range(n):
            while s and heights[s[-1]] >= heights[i]:
                top = s.pop()
                rightIdx[top] = i
            leftIdx[i] = s[-1] if s else -1
            s.append(i)
        ans = 0
        for i in range(n):
            tmp = (rightIdx[i] - leftIdx[i] - 1) * heights[i]
            ans = max(ans, tmp)
        return ans
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



### [768. Max Chunks To Make Sorted II](https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/)

**Example 2:**

```
Input: arr = [2,1,3,4,4]
Output: 4
Explanation:
We can split into two chunks, such as [2, 1], [3, 4, 4].
However, splitting into [2, 1], [3], [4], [4] is the highest number of chunks possible.
```

```c++
class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        // 2 1 3 9 10 8 
        // (1 2) 3 (8, 9, 10)  来一个，如果逆序了，就把最后几个缩点。
        stack<int> s;
        for (auto& x: arr) {
            if (s.empty() || x >= s.top()) {
                s.push(x);
                continue;
            }
            int head = s.top();
            while (!s.empty() && x < s.top()) {
                s.pop();
            }
            s.push(head);
        }
        return s.size();
     }
};
```



### [853. Car Fleet](https://leetcode.cn/problems/car-fleet/)

后面的车，如果追上了前面的车，就会和前面的车一样慢。问到达target前形成多少车队。

- 后面的车如果T <= 前一辆车，不考虑其他车的情况下，一定能缩点
  - 相遇前，后面的车不会比自己慢，但前面的车可能因为和更前面的车缩点而更慢
  - 所以一定会相遇
- T 单调下降则不会缩点

```python
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        # T: 5 4 3 2 1
        # T: 3 4 5 2 1
        posSpeedList = sorted(zip(position, speed))
        T = [(target - pos) / v for (pos, v) in posSpeedList]
        s = []
        for t in T:
            while (s and t >= s[-1]):
                s.pop()
            s.append(t)
        return len(s)
```

### [901. Online Stock Span](https://leetcode.cn/problems/online-stock-span/)

今天前连续<=今天价格的天数。单调递增栈，但是要维护每个团的size。

**Example 1:**

```
Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]

Explanation
StockSpanner stockSpanner = new StockSpanner();
stockSpanner.next(100); // return 1
stockSpanner.next(80);  // return 1
stockSpanner.next(60);  // return 1
stockSpanner.next(70);  // return 2 把60合并了，但是只记录70
stockSpanner.next(60);  // return 1
stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
stockSpanner.next(85);  // return 6
```

```python
class StockSpanner:

    def __init__(self):
        self.s = []

    def next(self, price: int) -> int:
        s = self.s
        cnt = 1
        while (s and price >= s[-1][0]):
            _, n = s.pop()
            cnt += n
        s.append((price, cnt))
        return cnt
```

或者记录 idx 

```python
class StockSpanner:

    def __init__(self):
        self.s = [(-1, float('inf'))]
        self.idx = -1
    def next(self, price: int) -> int:
        s = self.s
        while (s and price >= s[-1][1]):
            s.pop()
        self.idx += 1
        ans = self.idx - s[-1][0]
        s.append((self.idx, price))
        return ans
```

### [907. Sum of Subarray Minimums](https://leetcode.cn/problems/sum-of-subarray-minimums/)

**Example 1:**

```
Input: arr = [3,1,2,4]
Output: 17
Explanation: 
Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. 
Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.
Sum is 17.
```

```python
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        # 4 3 1 2 2
        #    [    ]
        #    [      ]
        s = []
        n = len(arr)
        left = [-1] * n
        right = [n] * n
        for i, num in enumerate(arr):
            while len(s) and num <= arr[s[-1]]:
                idx = s.pop()
                right[idx] = i
            left[i] = s[-1] if s else -1
            s.append(i)
 
        ans = 0
        m = 10 ** 9 + 7
        for i in range(n):
            ans += (right[i] - i) * (i - left[i]) * arr[i]
            ans %= m
        return ans
```





### [975. Odd Even Jump](https://leetcode.cn/problems/odd-even-jump/)　排序后的单调栈

左跳右边 比自己大中最小的

考虑单调上升的数字，但是索引却单调递减，他们在等待下一个数字可以pop他们

```python
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)

        incIdx = sorted(range(n), key=lambda i: (arr[i], i))
        oddNext = [i for i in range(n)]
        s = [] # 索引单调递减
        for i in incIdx:
            while s and i > s[-1]:
                oddNext[s.pop()] = i
            s.append(i)
        
        decIdx = sorted(range(n), key=lambda i: (-arr[i], i))
        evenNext = [i for i in range(n)]
        s = [] # 索引单调递减
        for i in decIdx:
            while s and i > s[-1]:
                evenNext[s.pop()] = i
            s.append(i)
        print(oddNext)
        print(evenNext)

        evenReach = [False] * (n - 1) + [True]
        oddReach = [False] * (n - 1) + [True]
        for i in range(n - 2, -1, -1):
            evenReach[i], oddReach[i] = oddReach[evenNext[i]], evenReach[oddNext[i]]
        return sum(oddReach)
```



### [1124. Longest Well-Performing Interval](https://leetcode.cn/problems/longest-well-performing-interval/)

a *tiring day* if and only if the number of hours worked is (strictly) greater than `8`.

A *well-performing interval* is an interval of days for which the number of tiring days > number of non-tiring days.

Return the length of the longest well-performing interval.

**Example 1:**

```
Input: hours = [9,9,6,0,6,6,9]
Output: 3
Explanation: The longest well-performing interval is [9,9,6].
```

```python
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        # 9 9 6 0 6 6 9
        # +1 +1 -1 -1 -1 -1 +1
        # 求最长的子数组，使得和 > 0
        # 那么记录前缀和s，要s[j] - s[i] > 0，找最远的i, j距离
        # 给定j，要i最远
        # 如果栈单调递减，来了一个j，那么pop出了它的i
        # 但是这些 i 还有用，还需要放回这个单调递减的栈。。。
        # 所以原始的pop只能解决最近的i, j距离

        # 思路回到寻找最长的一段i, j 使得 x[i] < x[j]
        # 只需要记录 原数组左边开始 单调递减的子序列，然后从右边开始不停pop它
        hours = [1 if x > 8 else -1 for x in hours]
        cumsum = [0] * (len(hours) + 1)
        for i in range(1, len(hours) + 1):
            cumsum[i] = cumsum[i - 1] + hours[i - 1]
 
        s = [0] # 左侧单调递减子序列 的坐标
        for i in range(1, len(cumsum)):
            if cumsum[i] < cumsum[s[-1]]:
                s.append(i)
        ans = 0
        for j in range(len(cumsum) - 1, -1, -1):
            while s and cumsum[j] > cumsum[s[-1]]:
                ans = max(ans, j - s.pop())
        return ans
```

[1574. Shortest Subarray to be Removed to Make Array Sorted](https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/)

Given an integer array `arr`, remove a subarray (can be empty) from `arr` such that the remaining elements in `arr` are **non-decreasing**.

Return *the length of the shortest subarray to remove*.

**Example 1:**

```
Input: arr = [1,2,3,(10,4,2),3,5]
Output: 3
Explanation: The shortest subarray we can remove is [10,4,2] of length 3. The remaining elements after that will be [1,2,3,3,5] which are sorted.
Another correct solution is to remove the subarray [3,10,4].
```

```python
class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)
        p1 = 0
        while p1 + 1 < n and arr[p1 + 1] >= arr[p1]:
            p1 += 1
        # [   p1] [delete] [j...]
        res = n - (p1 + 1) # 初始，右侧数组不存在，只保留 [0...p1]
        for p2 in range(n - 1, -1, -1):
            if p2 == p1:
                break
            while p1 >= 0 and arr[p1] > arr[p2]:
                p1 -= 1
            res = min(res, p2 - p1 - 1)
            if p2 - 1 >= 0 and arr[p2 - 1] > arr[p2]:
                break
        return res

```

### [1776. Car Fleet II](https://leetcode.cn/problems/car-fleet-ii/) (DP)

车往右开。求每个车合并的时间。不合并返回-1

```python
class Solution:
    def getCollisionTimes(self, cars: List[List[int]]) -> List[float]:
        # 只要后面的人速度大，一定能追上前一个
        # 但是追上的时候，前面那个车已经和之前的人合并了
        # 那么相当于在追之前的人

        # if v[2] < v[1], but 2 catches 1, 2 is actually catching the one before 1 who is very slow.

        # only cares about an exact catch. i.e. dS / dV is Time

        # start: car_n...car_1

        # key: (1) if I cannot catch someone, the person behind me cannot either
        #      (2) people behind me cannot change my speed.
        ans = [-1] * n

        def canExactCatch(i, j):
            # can i catch j?
            if cars[i][1] <= cars[j][1]:
                return 0
            t = (cars[j][0] - cars[i][0]) / (cars[i][1] - cars[j][1])
            if ans[j] == -1:
                return t
            else:
                if t < ans[j]:
                    return t
                else:
                    return 0

        s = [] # pop 不能追的人
        for i in range(n - 1, -1, -1):
            while s and not canExactCatch(i, s[-1]):
                s.pop()
            if s:
                ans[i] = canExactCatch(i, s[-1])
            s.append(i)

        return ans
```

### [1793. Maximum Score of a Good Subarray](https://leetcode.cn/problems/maximum-score-of-a-good-subarray/) 双指针

You are given an array of integers `nums` **(0-indexed)** and an integer `k`.

The **score** of a subarray `(i, j)` is defined as `min(nums[i], nums[i+1], ..., nums[j]) * (j - i + 1)`. A **good** subarray is a subarray where `i <= k <= j`.

Return *the maximum possible **score** of a **good** subarray.*

**Example 1:**

```
Input: nums = [1,4,3,7,4,5], k = 3
Output: 15
Explanation: The optimal subarray is (1, 5) with a score of min(4,3,7,4,5) * (5-1+1) = 3 * 5 = 15. 
```

```python
class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        
        n = len(nums)
        i, j = k, k
        currMin = nums[k]
        # while j <= n - 2 and nums[j + 1] >= currMin:
        #     j += 1
        # while i >= 1 and nums[i - 1] >= currMin:
        #     i -= 1
 
        ans = (j - i + 1) * currMin
 
        def getMoveDir(i, j):
            if i <= 0:
                return 1
            if j >= n - 1:
                return 0
            return nums[j + 1] >= nums[i - 1]
        
        while i >= 1 or j <= n - 2:
            if getMoveDir(i, j):
                # 向右 move
                j += 1
                currMin = min(currMin, nums[j])
                while j <= n - 2 and nums[j + 1] >= currMin:
                    j += 1
            else:
                i -= 1
                currMin = min(currMin, nums[i])
                while i >= 1 and nums[i - 1] >= currMin:
                    i -= 1
            ans = max(ans, (j - i + 1) * currMin)
 
        return ans

            
```

### [1856. Maximum Subarray Min-Product](https://leetcode.cn/problems/maximum-subarray-min-product/)

The **min-product** of an array is equal to the **minimum value** in the array **multiplied by** the array's **sum**.

- For example, the array `[3,2,5]` (minimum value is `2`) has a min-product of `2 * (3+2+5) = 2 * 10 = 20`.

```python
class Solution:
    def maxSumMinProduct(self, nums: List[int]) -> int:
        # 右边比自己严格小 (-1)
        # 左边比自己严格小 (n)
        n = len(nums)
        leftIdx = [-1] * n
        rightIdx = [n] * n
        s = []
        for i, num in enumerate(nums):
            while s and num < nums[s[-1]]:
                rightIdx[s.pop()] = i # 右侧严格
            if s:
                leftIdx[i] = s[-1] # 左侧不严格
            s.append(i)
        s = []
        cumsum = [0]
        for i in range(n):
            cumsum.append(cumsum[-1] + nums[i])
        m = 10**9 + 7
        ans = 0
        for i in range(n):
            ans = max(ans, (cumsum[rightIdx[i] - 1 + 1] - cumsum[leftIdx[i] + 1]) * nums[i])
        return ans % m
```



### [1944. Number of Visible People in a Queue](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/)

A person can **see** another person to their right in the queue if everybody in between is **shorter** than both of them.

```
Input: heights = [5,1,2,3,10]
Output: [4,1,1,1,0]
```

```python
class Solution:
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        #    3 5 7 10 (无法看到再之前的人)
        # 6 /  / 7 10 (被6pop的3和5反正被6挡住了, 不能被6左边的人看到)
        n = len(heights)
        ans = [0] * n
        s = []
        for i in range(n - 1, -1, -1):
            while s and heights[i] > heights[s[-1]]:
                s.pop()
                ans[i] += 1
            if s:
                ans[i] += 1
            s.append(i)
        return ans
```

### [1950. Maximum of Minimum Values in All Subarrays](https://leetcode.cn/problems/maximum-of-minimum-values-in-all-subarrays/)

You are given an integer array `nums` of size `n`. You are asked to solve `n` queries for each integer `i` in the range `0 <= i < n`.

To solve the `ith` query:

1. Find the **minimum value** in each possible subarray of size `i + 1` of the array `nums`.
2. Find the **maximum** of those minimum values. This maximum is the **answer** to the query.

Return *a **0-indexed** integer array* `ans` *of size* `n` *such that* `ans[i]` *is the answer to the* `ith` *query*.

A **subarray** is a contiguous sequence of elements in an array.

```python
class Solution:
    def findMaximums(self, nums: List[int]) -> List[int]:
        # 1 (3 5) 2
        # the right idx is determined when an element is popped
        # left idx is just the element left to me (or -1)
        # finally, 1 2 3 4 5, the right index is n
        d = {} # length -> max of min
        # 5 5 5 6 7     5:5 contains 4:5 3:5.... 1:5
        #   5 5 6 7
 
        # 0 1 1 2 大于等于就pop也没关系
        n = len(nums)
        s = [-1]
        for i in range(n):
            while len(s) > 1 and nums[i] <= nums[s[-1]]:
                head = s.pop()
                cnt = i - s[-1] - 1
                d[cnt] = max(d.get(cnt, -1), nums[head])
            s.append(i)
        
        for j in range(len(s) - 1, 0, -1):
            cnt = n - s[j - 1] - 1
            d[cnt] = max(d.get(cnt, -1), nums[s[j]])
        
        ans = [0] * n
        ans[n - 1] = d[n]
        for i in range(n - 2, -1, -1):
            ans[i] = max(ans[i + 1], d.get(i + 1, -1))
        return ans
```

### [1996. The Number of Weak Characters in the Game](https://leetcode.cn/problems/the-number-of-weak-characters-in-the-game/)

A character is said to be **weak** if any other character has **both** attack and defense levels **strictly greater** than this character's attack and defense levels.

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:

        # 1 3 / 1 3/ 1 2 / 3 6 / 5 5/ 6 3
        # 先按照a[0]排序 (a[0], -a[1])
        # 然后看a[1], 右边第一个比自己严格大的在不在? 被 pop -> weak

        # 也不需要单调栈
        ans = 0
        properties.sort(key=lambda x: (x[0], -x[1]))
        maxR = properties[-1][1]
        for i in range(len(properties) - 2, -1, -1):
            if maxR > properties[i][1]:
                ans += 1
            else:
                maxR = properties[i][1]
        return ans
 
        # 单调栈:
        properties.sort(key=lambda x: (x[0], -x[1]))
        res = 0
        s = []
        for x in properties:
            while s and x[1] > s[-1]:
                s.pop()
                res += 1
            s.append(x[1])
        return res
```

```c++
class Solution {
public:
    int numberOfWeakCharacters(vector<vector<int>>& properties) {
        // 按第一维从大到小排序, 第二维从小到大
        sort(
            properties.begin(),
            properties.end(),
            [] (const vector<int> & a, const vector<int> & b) {
                return a[0] == b[0] ? (a[1] < b[1]) : (a[0] > b[0]);
            }
        );

        int maxSecond = -1;
        int ans = 0;
        for (const auto & p: properties) {
            if (maxSecond > p[1]) {
                ++ans;
            } else {
                maxSecond = p[1];
            }
        }
        return ans;
    }
};
```



### [2104. Sum of Subarray Ranges](https://leetcode.cn/problems/sum-of-subarray-ranges/)

The **range** of a subarray is the difference between the largest and smallest element in the subarray.

Return *the **sum of all** subarray ranges of* `nums`*.*

一个数字是多少个子数组的最值?

```python
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        # [1] 3 5 (2) 3 [1]
        # 0   1 2  3  4  5

        # 1 1
        #-1,1 ->1
        #-1,2 -> 2 * 1

        # count: min for how many intervals?
        # inc stack, left is left bound, be popped by right bound

        # count: max for how many intervals?
        # dec stack, left is left bound, be popped by left bound

        def f(compareFunc):
            res = 0
            s = [-1]
            for i in range(len(nums)):
                while len(s) > 1 and compareFunc(nums[i], nums[s[-1]]):
                    idx = s.pop()
                    left = s[-1]
                    right = i
                    res += (idx - left) * (right - idx) * nums[idx]
                s.append(i)
            for i in range(1, len(s)):
                left = s[i - 1]
                right = len(nums)
                idx = s[i]
                res += (idx - left) * (right - idx) * nums[idx]
            return res
 
        cntMin = f(lambda *args: args[0] < args[1])
        cntMax = f(lambda *args: args[0] > args[1])
        return cntMax - cntMin

```





### [2281. Sum of Total Strength of Wizards](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/)

For a **subarray** of `strength`， the **total strength** is defined as the **product** of the following two values:

- The strength of the **weakest** wizard in the group.
- The **total** of all the individual strengths of the wizards in the group.

```python
class Solution:
    def totalStrength(self, strength: List[int]) -> int:
        n = len(strength)
        # prefix sum
        s = [0] * n
        s[0] = strength[0]
        for i in range(1, n):
            s[i] = s[i - 1] + strength[i]
        
        # 前缀和的前缀和
        ss = [0] * n
        ss[0] = s[0]
        for i in range(1, n):
            ss[i] = ss[i - 1] + s[i]

        res = 0
        s = [-1]

        def calcValue(left, right):
            v1 = ss[right]
            v2 = ss[idx - 1] if idx - 1 >= 0 else 0
            v3 = ss[left - 2] if left - 2 >= 0 else 0
            return (idx - left + 1) * (v1 - v2) - (right - idx + 1) * (v2 - v3)

        for i in range(n):
            while len(s) > 1 and strength[i] <= strength[s[-1]]:
                idx = s.pop()
                left = s[-1] + 1
                right = i - 1
                res += calcValue(left, right) * strength[idx]
            s.append(i)

        print(s, res)
        
        # 还剩下单调递增栈
        m = 10 ** 9 + 7
        for i in range(1, len(s)):
            idx = s[i]
            left = s[i - 1] + 1
            right = n - 1
            res += calcValue(left, right) * strength[idx]
            res = res % m

        return res
```

### [2282. Number of People That Can Be Seen in a Grid](https://leetcode.cn/problems/number-of-people-that-can-be-seen-in-a-grid/)

可以看到右边/下方的人，要求中间人都严格小于我们。

```python
class Solution:
    def seePeople(self, heights: List[List[int]]) -> List[List[int]]:
        m, n = len(heights), len(heights[0])
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            s = []
            for j in range(n - 1, -1, -1):
                while s and heights[i][j] > heights[i][s[-1]]:
                    s.pop()
                    ans[i][j] += 1
                if s:
                    if heights[i][j] == heights[i][s[-1]]:
                        s.pop()
                    ans[i][j] += 1
                s.append(j)
        for j in range(n):
            s = []
            for i in range(m - 1, -1, -1):
                while s and heights[i][j] > heights[s[-1]][j]:
                    s.pop()
                    ans[i][j] += 1
                if s:
                    if heights[i][j] == heights[s[-1]][j]:
                        s.pop()
                    ans[i][j] += 1
                s.append(i)
        return ans
```




### [2289. Steps to Make Array Non-decreasing](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/) 难

In one step, **remove** all elements `nums[i]` where `nums[i - 1] > nums[i]`

真的会让step增加的是  7 (41) (51) (61) 这样的序列. 第一步之后, 还有 7 5 6。7 .. 5 6

7有两步数才到5, 然后有3步才到6. 最后7的T=3. 如果7左边有10, 尽管10有一步到7, 但是max(1, 3)=3. 

5 4 3 2 1只需要一步就只剩下5了

```python
class Solution:
    def totalSteps(self, nums: List[int]) -> int:
        # 12 7 (5 2 1) (6 3 2)
        #       T=1      T=1 
        # 对于  7 5(1) 6(1)
        #        max(1,1)=1 max(1,2)=2
        # 12 7(T=2)
        # max(1,2)=2

        s = [] # 存(value, t)
        res = 0
        for i in range(len(nums) - 1, -1, -1):
            num = nums[i]
            # pop ->
            t = 0
            while s and num > s[-1][0]:
                _, t0 = s.pop()
                t += 1
                t = max(t, t0)
            res = max(res, t)
            s.append((num, t))
        return res
```

### [2355. Maximum Number of Books You Can Take](https://leetcode.cn/problems/maximum-number-of-books-you-can-take/)

从子数组拿书，但是书必须严格单调递增。找到第一次break等差数列的地方。

**Example 1:**

```
Input: books = [8,5,2,7,9]
Output: 19
Explanation:
- Take 1 book from shelf 1.
- Take 2 books from shelf 2.
- Take 7 books from shelf 3.
- Take 9 books from shelf 4.
You have taken 19 books, so return 19.
It can be proven that 19 is the maximum number of books you can take.
```

```python
class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        # arr[i] arr[i]-1   arr[i]-2
        #    i       i-1     i-2
        # arr[j] < arr[i] - (i - j) 需要停止
        # arr[j] - j < arr[i] - i
        # 从j开始是独立的较小的书本，已经求过 (-1)
        # 不然就是-1等差数列  j + 1 -> i
        #                             num[i]... nums[i] - (i-j+1)
        n = len(books)
        nums = [books[i] - i for i in range(n)]
    
        # 求左边第一个比自己严格小的
        leftIdx = [-1] * n
        s = [-1]
        for i in range(n):
            while len(s) > 1 and nums[i] <= nums[s[-1]]:
                s.pop()
            leftIdx[i] = s[-1]
            s.append(i)
 
        dp = [0] * n
        for i in range(n):
            j = leftIdx[i]
            # from j+1 -> i
            up = books[i]
            down = max(books[i] - (i - (j + 1)), 0)
            dp[i] = (up + down) * (up - down + 1) // 2
            if j != -1:
                dp[i] += dp[j]
 
        return max(dp)
```

### [2454. Next Greater Element IV](https://leetcode.cn/problems/next-greater-element-iv/) 优先队列

右边第二个比自己打的数字

```python
class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n
        s = []
        q = [] # (value, idx) 从小到大排 
               # cache 那些已经被pop一次的数字，等待pop第二次
        for i in range(n):
            while q and nums[i] > q[0][0]:
                res[q[0][1]] = nums[i]
                heapq.heappop(q)
            while s and nums[i] > nums[s[-1]]:
                heapq.heappush(q, (nums[s[-1]], s[-1]))
                s.pop()
            s.append(i)
        return res
```

### [2617. Minimum Number of Visited Cells in a Grid](https://leetcode.cn/problems/minimum-number-of-visited-cells-in-a-grid/) 二维优先队列

原点出发，每次可以向右或者下方走最多`grid[i][j]`格。求最短路径。

```python
class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1:
            return 1
        
        colQ = [[] for _ in range(n)]
        for i in range(m):
            rowQ = []
            for j in range(n):
                if i == 0 and j == 0:
                    heapq.heappush(rowQ, (1, j))
                    heapq.heappush(colQ[j], (1, i))
                else:
                    # 从左边来
                    tmp = float('inf')
                    while rowQ:
                        step, y = rowQ[0]
                        if y + grid[i][y] >= j:
                            tmp = min(tmp, step + 1)
                            break
                        else:
                            heapq.heappop(rowQ)
                    # 从上面来
                    while colQ[j]:
                        step, x = colQ[j][0]
                        if x + grid[x][j] >= i:
                            tmp = min(tmp, step + 1)
                            break
                        else:
                            heapq.heappop(colQ[j])
                    if tmp != float('inf'):
                        heapq.heappush(rowQ, (tmp, j))
                        heapq.heappush(colQ[j], (tmp, i))
                if i == m - 1 and j == n - 1:
                    return tmp if tmp != float('inf') else -1
        return -1
```

### [2736. Maximum Sum Queries](https://leetcode.cn/problems/maximum-sum-queries/)　Hard

You are given two **0-indexed** integer arrays `nums1` and `nums2`, each of length `n`, and a **1-indexed 2D array** `queries` where `queries[i] = [xi, yi]`.

For the `ith` query, find the **maximum value** of `nums1[j] + nums2[j]` among all indices `j` `(0 <= j < n)`, where `nums1[j] >= xi` and `nums2[j] >= yi`, or **-1** if there is no `j` satisfying the constraints.



1. x 要从大到小看，这样x永远不是bottleneck

2. (y, sum) : x变小, y变小, 丢弃; x变小, y变大, sum可能大也可能小

   如果 y变大, sum变大, 那么丢弃之前的 stack

   所以 stack 中, y变大, sum一定变小.

   给定要求的y_min, 只需要找到y_min的lower_bound.

```python
class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # nums中，相同的x，只保留y大的
        # x减小，必须有y增加，才考虑
        # (y, sum) < (y', sum') < ...那么sum' < sum 不然(y, sum)被pop
        # 新的qeury, x都满足条件了, y lower bound
        nums = sorted([(x, y) for x, y in zip(nums1, nums2)], key=lambda x: -x[0])
        queries = sorted([(q[0], q[1], i) for i, q in enumerate(queries)], key=lambda x: -x[0])
        n = len(queries)
        ans = [-1] * n
 
        s = []
        j = 0
        for q0, q1, qIdx in queries:
            # 更新 stack
            while j < len(nums) and nums[j][0] >= q0:
                # update
                x, y = nums[j]
                while s and x + y >= s[-1][1]: # 隐含了 y >= s[-1][0]
                    s.pop()
                if not s or y > s[-1][0]: # 此时 必有 新sum 较小
                    s.append((y, x + y))
                j += 1
            k = bisect.bisect_left(s, (q1, 0))
            if k < len(s):
                ans[qIdx] = s[k][1]
        return ans
```

### [2818. Apply Operations to Maximize Score](https://leetcode.cn/problems/apply-operations-to-maximize-score/)

元素的真实value是它prime factor的个数。要用到pow(x,y,mod)

start with a score of `1`. You have to maximize your score by applying the following operation at most `k` times:

- Choose any **non-empty** subarray `nums[l, ..., r]` that you haven't chosen previously.
- Choose an element `x` of `nums[l, ..., r]` with the highest **prime score**. If multiple such elements exist, choose the one with the smallest index.
- Multiply your score by `x`.

The **prime score** of an integer `x` is equal to the number of distinct prime factors of `x`. For example, the prime score of `300` is `3` since `300 = 2 * 2 * 3 * 5 * 5`.

Return *the **maximum possible score** after applying at most* `k` *operations*.

Since the answer may be large, return it modulo `10^9 + 7`.

```python
class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        # num of factors
        # prime -> 0
        MAXN = max(2, max(nums))
        nFact = [0] * (MAXN + 1)
        for i in range(2, MAXN + 1):
            if nFact[i] == 0:
                for j in range(i, MAXN + 1, i):
                    nFact[j] += 1
        # print([nFact[i] for i in nums])

        # 左边 >= 自己，右边严格大于自己的数字
        # （不严格）单调递减栈
        n = len(nums)
        leftIdx = [-1] * n
        rightIdx = [n] * n
        s = []
        for i in range(n):
            while s and nFact[nums[i]] > nFact[nums[s[-1]]]:
                rightIdx[s.pop()] = i
            if s:
                leftIdx[i] = s[-1]
            s.append(i)
        # 还剩下一些，他们的右边界已经是n，左边界都在加入时更新了
 
        sortedNums = sorted([(num, i) for i, num in enumerate(nums)], key=lambda x: -x[0])
        ans = 1
        m = 10 ** 9 + 7
 
        for num, idx in sortedNums:
            tmp = (idx - leftIdx[idx]) * (rightIdx[idx] - idx)
            tmp = min(tmp, k) 
            ans = (ans * pow(num, tmp, m)) % m
            k -= tmp
            if k == 0:
                break
        return ans
```

 

### [2863. Maximum Length of Semi-Decreasing Subarrays](https://leetcode.cn/problems/maximum-length-of-semi-decreasing-subarrays/)

You are given an integer array `nums`.

Return *the length of the **longest semi-decreasing** subarray of* `nums`*, and* `0` *if there are no such subarrays.*

- A non-empty array is **semi-decreasing** if its first element is **strictly greater** than its last element.

**Example 1:**

```
Input: nums = [7,6,5,4,3,2,1,6,10,11]
Output: 8
Explanation: Take the subarray [7,6,5,4,3,2,1,6].
```

```python
class Solution:
    def maxSubarrayLength(self, nums: List[int]) -> int:
        # 6 5 11 4 10 5 8
        s = [0]
        for i in range(1, len(nums)):
            if nums[i] > nums[s[-1]]:
                s.append(i)
        res = 0
        for j in range(len(nums) - 1, -1, -1):
            while s and nums[s[-1]] > nums[j]:
                idx = s.pop() # pop 表示ok
                res = max(res, j - idx + 1)
        return res
        
        # # 7 6 7 6 5 4
        # # 10.(10). [7(2)] 7[0] 6(1) 6(3) 5 4 
        # sortedNums = sorted([(num, i) for i, num in enumerate(nums)], key=lambda x: (-x[0], -x[1]))
        # maxNum, maxIdx = sortedNums[0]
        # ans = 0
        # for num, idx in sortedNums[1:]:
        #     if num >= maxNum or idx < maxIdx:
        #         maxNum = num
        #         maxIdx = idx
        #     else:
        #         ans = max(ans, idx - maxIdx + 1)
        # return ans
```

 

### [2865. Beautiful Towers I](https://leetcode.cn/problems/beautiful-towers-i/)

remove some bricks to form a **mountain-shaped** tower arrangement. In this arrangement, the tower heights are non-decreasing, reaching a maximum peak value with one or multiple consecutive towers and then non-increasing.

Return the **maximum possible sum** of heights of a mountain-shaped tower arrangement.

```python
class Solution:
    def maximumSumOfHeights(self, heights: List[int]) -> int:
        # 找到左侧最近的 idx, h[idx] <= h[i]
        # left[idx] + (idx+1 -> i-1) * h[i]
        n = len(heights)
        left = [0] * n
        s = []
        for i in range(n):
            while s and heights[i] < heights[s[-1]]:
                s.pop()
            idx = s[-1] if s else -1
            left[i] = (i - 1 - (idx + 1) + 1 + 1) * heights[i]
            if idx != -1:
                left[i] += left[idx]
            s.append(i)
        
        # 单调递增

        # 找到右侧最近的 idx, h[idx] <= h[i]
        # left[idx] + (i+1 -> idx-1) * h[i]
        right = [0] * n
        s = []
        for i in range(n - 1, -1, -1):
            while s and heights[i] < heights[s[-1]]:
                s.pop()
            idx = s[-1] if s else n
            right[i] = (idx - 1 - (i + 1) + 1 + 1) * heights[i]
            if idx != n:
                right[i] += right[idx]
            s.append(i)

        res = 0
        for i in range(n):
            res = max(res, left[i] + right[i] - heights[i])
        return res
```

 

### [3113. Find the Number of Subarrays Where Boundary Elements Are Maximum](https://leetcode.cn/problems/find-the-number-of-subarrays-where-boundary-elements-are-maximum/)

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        # 2 (4)_1 3 2 (4)_1 3 (4) 5 3 (4) 8 5
        # right, first >= me position (1) == ok (2) not ok.
        # can extend to the next
        # me as the start, next me -> next next me.... end
        # save the cnt
        # 左边第一个>=自己的数字
        n = len(nums)
        cnt = [0] * n
        leftIdx = [-1] * n
        s = []
        ans = 0
        for i, num in enumerate(nums):
            while s and num > nums[s[-1]]:
                s.pop()
            # num <= stack
            if s:
                leftIdx[i] = s[-1]
                if nums[leftIdx[i]] == num:
                    cnt[i] = cnt[leftIdx[i]] + 1
                    ans += cnt[i]
            s.append(i)
        return ans + len(nums)
```

### [面试题 16.16. 部分排序](https://leetcode.cn/problems/sub-sort-lcci/)

至少sort中间的一部分

```python
class Solution:
    def subSort(self, array: List[int]) -> List[int]:
        # 1 2 3 4 [2] 6 10 [3] 15 19
        currMax = float('-inf')
        rightIdx = -1
        for i in range(len(array)):
            if array[i] >= currMax:
                currMax = array[i]
            else:
                # 出现逆序对
                rightIdx = i
        currMin = float('inf')
        leftIdx = -1
        for i in range(len(array) - 1, -1, -1):
            if array[i] <= currMin:
                currMin = array[i]
            else:
                leftIdx = i
        return [leftIdx, rightIdx]
```

### [面试题 17.21. 直方图的水量](https://leetcode.cn/problems/volume-of-histogram-lcci/)

柱子的位置和高度，能接多少雨水。单调递减栈。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        s = []
        ans =0 
        for i, h in enumerate(height):
            while s and h >= height[s[-1]]:
                idx = s.pop()
                if len(s):
                    leftIdx = s[-1]
                    w = i - leftIdx - 1
                    add = min(height[leftIdx] - height[idx], h - height[idx])
                    ans += w * add
            s.append(i)
        return ans
```


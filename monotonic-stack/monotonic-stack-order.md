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


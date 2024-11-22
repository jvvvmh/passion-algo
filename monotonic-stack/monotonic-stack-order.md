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

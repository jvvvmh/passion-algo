# 栈

[TOC]

## 单调栈

### [42. Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)

<img src="images/rain.png" alt="image" style="zoom:38%;" />

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




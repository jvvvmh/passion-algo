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


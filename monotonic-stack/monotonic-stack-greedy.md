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




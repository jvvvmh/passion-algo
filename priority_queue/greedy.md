Priority Queue (Greedy)

[TOC]



#### [502. IPO](https://leetcode.cn/problems/ipo/)

Start with capital w, at most do k projects, after doing each project, the profit can update w. What is the final w?

Among all the affordable projects now, always do the most profitable one, and you have higher capital.

Queue: profit (high) from affordable projects. Update when you have more capital.

```C++
typedef pair<int, int> pi;

class Solution {
public:
    int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
        vector<pi> x;
        int n = profits.size();
        for (int i = 0; i < n; ++i) {
            x.push_back(pair(profits[i], capital[i]));
        }
        sort(x.begin(), x.end(), [](pi a, pi b) {return a.second < b.second;});
        priority_queue<int, vector<int>, less<int>> q;
        int idx = 0;
        int cnt = 0;
        while (cnt < k) {
            while (idx < n && x[idx].second <= w) {
                q.push(x[idx].first);
                idx += 1;
            }
            if (q.empty()) {
                return w;
            }
            int head = q.top();
            q.pop();
            w += head;
            cnt += 1;
        }
        return w;
    }
};
```


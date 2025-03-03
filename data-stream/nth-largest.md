





#### [295. Find Median from Data Stream](https://leetcode.cn/problems/find-median-from-data-stream/)

```c++
class MedianFinder {
public:
    priority_queue<int, vector<int>, less<int>> qLeft;       // top is large
    priority_queue<int, vector<int>, greater<int>> qRight;   // top is small

    MedianFinder() {}
    
    template <typename T1, typename T2>
    void adjust(T1 & q1, T2 & q2) {
        int tmp = q1.top();
        q1.pop();
        q2.push(tmp);
    }
 
    void addNum(int num) {
        if (qRight.empty() || num >= qRight.top()) {
            qRight.push(num);
        } else {
            qLeft.push(num);
        }
        if (qRight.size() > qLeft.size() + 1) {
            adjust(qRight, qLeft);
        } else if (qLeft.size() > qRight.size()) {
            adjust(qLeft, qRight);
        }
    }
    
    double findMedian() {
        if (qLeft.size() == qRight.size()) {
            return (qLeft.top() + qRight.top()) / 2.;
        }
        return qRight.top();
    }
};
```



[480. Sliding Window Median](https://leetcode.cn/problems/sliding-window-median/)

```c++
vector<double> f(vector<int> & nums, int w, int k) {
    // initial k largert nums
    vector<double> res;
    multiset<int> s;
    for (int i = 0; i < w; ++i) {
        s.insert(nums[i]);
    }
    auto ptr = s.begin();
    for (int i = 0; i < k - 1; ++i) {
        ++ptr;
    }
    res.push_back(*ptr);
    for (int i = w; i < nums.size(); ++i) {
        if (nums[i] < *ptr) {
            s.insert(nums[i]);
        } else{
            s.insert(nums[i]);
            ++ptr; 
        }
        
        if (s.find(nums[i - w]) == ptr) {
            auto tmp = ptr;
            ptr--;
            s.erase(tmp);
        } else {
            if (nums[i - w] <= *ptr) {
                s.erase(s.find(nums[i - w]));
            } else {
                --ptr;
                s.erase(s.find(nums[i - w]));
            }
        }
        res.push_back(*ptr);
    }
    return res;
}

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        if (k & 1) {
            return f(nums, k, k / 2 + 1);
        }
        auto v1 = f(nums, k, k / 2);
        auto v2 = f(nums, k, k / 2 + 1);
        vector<double> res(v1.size());
        for (int i = 0; i < v1.size(); ++i) {
            res[i] = (v1[i] + v2[i]) / 2.;
        }
        return res;
    }
};
```


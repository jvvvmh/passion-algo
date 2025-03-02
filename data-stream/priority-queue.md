





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


 list 是顺序容器的一种。list 是一个双向链表。使用 list 需要包含头文件 list。双向链表的每个元素中都有一个[指针](http://c.biancheng.net/c/80/)指向后一个元素，也有一个指针指向前一个元素 



 list 容器不支持根据下标随机存取元素。 



| 成员函数或成员函数模板                                       | 作  用                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| void push_front(const T & val)                               | 将 val 插入链表最前面                                        |
| void pop_front()                                             | 删除链表最前面的元素                                         |
| void sort()                                                  | 将链表从小到大排序                                           |
| void remove (const T & val)                                  | 删除和 val 相等的元素                                        |
| remove_if                                                    | 删除符合某种条件的元素                                       |
| void unique()                                                | 删除所有和前一个元素相等的元素                               |
| void merge(list <T> & x)                                     | 将链表 x 合并进来并清空 x。要求链表自身和 x 都是有序的       |
| void splice(iterator i, list <T> & x, iterator first, iterator last) | 在位置 i 前面插入链表 x 中的区间 [first, last)，并在链表 x 中删除该区间。链表自身和链表 x 可以是同一个链表，只要 i 不在 [first, last) 中即可 |

很多重载的没有展示



 STL 中的算法 sort 可以用来对 vector 和 deque 排序，它需要随机访问迭代器的支持。

因为 list 不支持随机访问迭代器，所以不能用算法 sort 对 list 容器排序。

因此，list 容器引入了 sort 成员函数以完成排序。 



```c++
#include <list>  //使用 list 需要包含此头文件
#include <iostream>
#include <algorithm>  //使用STL中的算法需要包含此头文件
using namespace std;

class A {
private: int n;
public:
    A(int n_) { n = n_; }
    friend bool operator < (const A & a1, const A & a2);
    friend bool operator == (const A & a1, const A & a2);
    friend ostream & operator << (ostream & o, const A & a);
};

bool operator < (const A & a1, const A & a2) {
    return a1.n < a2.n;
}

bool operator == (const A & a1, const A & a2) {
    return a1.n == a2.n;
}

ostream & operator << (ostream & o, const A & a) {
    o << a.n;
    return o;
}

template <class T>
void Print(T first, T last)
{
    for (; first != last; ++first)
        cout << *first << " ";
    cout << endl;
}

int main()
{
    A a[5] = { 1, 3, 2, 4, 2 };
    A b[7] = { 10, 30, 20, 30, 30, 40, 40 };
    
    list<A> lst1(a, a + 5), lst2(b, b + 7);
    lst1.sort();
    cout << "1)"; Print(lst1.begin(), lst1.end());  //输出：1)1 2 2 3 4
    
    lst1.remove(2);  //删除所有和A(2)相等的元素
    cout << "2)"; Print(lst1.begin(), lst1.end());  //输出：2)1 3 4
    
    lst2.pop_front();  //删除第一个元素	10
    cout << "3)"; Print(lst2.begin(), lst2.end());  //输出：3)30 20 30 30 40 40
    
    lst2.unique();  //删除所有和前一个元素相等的元素
    cout << "4)"; Print(lst2.begin(), lst2.end());  //输出：4)30 20 30 40
    
    lst2.sort();
    lst1.merge(lst2);  //合并 lst2 到 lst1 并清空 lst2
    cout << "5)"; Print(lst1.begin(), lst1.end());  //输出：5)1 3 4 20 30 30 40
    cout << "6)"; Print(lst2.begin(), lst2.end());  //lst2是空的，输出：6)
    
    lst1.reverse();  //将 lst1 前后颠倒
    cout << "7)"; Print(lst1.begin(), lst1.end());  //输出 7)40 30 30 20 4 3 1
    
    lst2.insert(lst2.begin(), a + 1, a + 4);  //在 lst2 中插入 3,2,4 三个元素
    list <A>::iterator p1, p2, p3;
    p1 = find(lst1.begin(), lst1.end(), 30);
    p2 = find(lst2.begin(), lst2.end(), 2);
    p3 = find(lst2.begin(), lst2.end(), 4);
    lst1.splice(p1, lst2, p2, p3);  //将[p2, p3)插入p1之前，并从 lst2 中删除[p2,p3)
    cout << "8)"; Print(lst1.begin(), lst1.end());  //输出：8)40 2 30 30 20 4 3 1
    cout << "9)"; Print(lst2.begin(), lst2.end());  //输出：9)3 4
    
    return 0;
}
```


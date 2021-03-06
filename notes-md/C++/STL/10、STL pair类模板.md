 在学习关联容器之前，首先要了解 [STL](http://c.biancheng.net/stl/) 中的 pair 类模板，因为关联容器的一些成员函数的返回值是 pair 对象，而且 map 和 multimap 容器中的元素都是 pair 对象。pair 的定义如下： 

```c++
template <class_Tl, class_T2>
struct pair
{
    _T1 first;
    _T2 second;
    pair(): first(), second() {}  //用无参构造函数初始化 first 和 second
    pair(const _T1 &__a, const _T2 &__b): first(__a), second(__b) {}
    template <class_U1, class_U2>
    pair(const pair <_U1, _U2> &__p): first(__p.first), second(__p.second) {}
};
```

 pair实例化出来的类都有两个成员变量，一个是 first, 一个是 second。

STL 中还有一个函数模板 make_pair，其功能是生成一个 pair 模板类对象。make_pair 的源代码如下： 

```c++
template <class T1, class T2>
pair<T1, T2 > make_pair(T1 x, T2 y)
{
    return ( pair<T1, T2> (x, y) );
}
```

 pair 模板中的第三个构造函数是函数模板，参数必须是一个 pair 模板类对象的引用。程序中第 9 行的 p3 就是用这个构造函数初始化的。 



 下面的程序演示了 pair 和 make_pair 的用法。 

```C++
#include <iostream>
using namespace std;
int main()
{
    pair<int,double> p1;
    cout << p1.first << "," << p1.second << endl; //输出  0,0   
    
    pair<string,int> p2("this",20);
    cout << p2.first << "," << p2.second << endl; //输出  this,20
    
    pair<int,int> p3(pair<char,char>('a','b'));
    cout << p3.first << "," << p3.second << endl; //输出  97,98
    
    pair<int,string> p4 = make_pair(200,"hello");
    cout << p4.first << "," << p4.second << endl; //输出  200,hello
    
    return 0;
}
```


# Leetcode 题解 - 字符串
<!-- GFM-TOC -->
* [Leetcode 题解 - 字符串](#leetcode-题解---字符串)
    * [1. 字符串循环移位包含](#1-字符串循环移位包含)
    * [2. 字符串循环移位](#2-字符串循环移位)
    * [3. 字符串中单词的翻转](#3-字符串中单词的翻转)
    * [4. 两个字符串包含的字符是否完全相同](#4-两个字符串包含的字符是否完全相同)
    * [5. 计算一组字符集合可以组成的回文字符串的最大长度](#5-计算一组字符集合可以组成的回文字符串的最大长度)
    * [6. 字符串同构](#6-字符串同构)
    * [7. 回文子字符串个数](#7-回文子字符串个数)
    * [8. 判断一个整数是否是回文数](#8-判断一个整数是否是回文数)
    * [9. 统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数](#9-统计二进制字符串中连续-1-和连续-0-数量相同的子字符串个数)
<!-- GFM-TOC -->


## 1. 循环移位包含

```html
[编程之美 3.1](#)	s1 = AABCD, s2 = CDAA	Return : true
```

给定两个字符串 s1 和 s2，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。

s1 进行循环移位的结果是 s1+s1 的子字符串，因此只要判断 s2 是否是 s1+s1 的子字符串即可。

## 2. 循环移位

```html
将字符串向右循环移动 k 位。   s = "abcd123" k = 3		Return "123abcd"
```

将 abcd123 中的 abcd 和 123 **单独翻转**，得到 dcba321，然后对**整个字符串进行翻转**，得到 123abcd。

## 3. 单词的翻转

```html
s = "I am a student"	Return "student a am I"		将每个单词翻转，然后将整个字符串翻转。
```

## 4. 判断字符排列

简单： [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

```html
s = "anagram", t = "nagaram", return true.		s = "rat", t = "car", return false.
```

可以用 HashMap 来映射字符与出现次数，然后比较两个字符串出现的字符数量是否相同。由于本题的字符串只包含 26 个小写字符，因此可以使用长度为 26 的整型数组对字符串出现的字符进行统计，不再使用 HashMap。

```java
public boolean isAnagram(String s, String t) {
    int[] cnts = new int[26];
    for (char c : s.toCharArray()) { cnts[c - 'a']++; }
    for (char c : t.toCharArray()) { cnts[c - 'a']--; }
    for (int cnt : cnts) { if (cnt != 0) return false; }
    return true;
}
```

## 5. 最长回文串

简单：[409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/)

```html
输入:"abccccdd"	输出:7	解释:可以构造的最长的回文串是"dccaccd", 长度是 7。
构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
```

使用长度为 256 的整型数组来统计每个字符出现的个数，每个字符有偶数个可以用来构成回文字符串。

因为回文字符串最中间的那个字符可以单独出现，所以如果有单独的字符就把它放到最中间。

```java
public int longestPalindrome(String s) {
    int[] cnts = new int[256];
    for (char c : s.toCharArray()) cnts[c]++;			//计数
    int palindrome = 0;
    for (int cnt : cnts) palindrome += (cnt / 2) * 2;
    // 这个条件下 s 中一定有单个未使用的字符存在，可以把这个字符放到回文的最中间
    if (palindrome < s.length()) palindrome++;
    return palindrome;
}
```

```c
int longestPalindrome(string s) {
    unordered_map<char, int> count;
    int ans = 0;
    for (char c : s) ++count[c];
    for (auto p : count) {
        int v = p.second;
        ans += v / 2 * 2;
        if (v % 2 == 1 and ans % 2 == 0) ++ans; //最多只会有一次的情况，后边再有奇数的不再自增
    }
    return ans;
}
```

## 6. 同构

简单： [205. 同构字符串](https://leetcode-cn.com/problems/isomorphic-strings/)

```html
输入： "paper", "title" 输出：true  	输入："foo", "bar"输出：false  
```

记录一个字符上次出现的位置，如果两个字符串中的字符上次出现的位置一样，那么就属于同构。

```java
public boolean isIsomorphic(String s, String t) {
    int[] preIndexOfS = new int[256];
    int[] preIndexOfT = new int[256];
    for (int i = 0; i < s.length(); i++) {
        char sc = s.charAt(i), tc = t.charAt(i);
        if (preIndexOfS[sc] != preIndexOfT[tc]) return false;
        preIndexOfS[sc] = i + 1; // 注意是 i + 1，+2也行，为了处理 "ab" "aa" 的结果是 false
        preIndexOfT[tc] = i + 1;
    }
    return true;
}
```

```c
bool isIsomorphic(string s, string t) {
    unordered_map<char, char> s2t;
    unordered_map<char, char> t2s;
    int len = s.length();
    for (int i = 0; i < len; ++i) {
        char x = s[i], y = t[i];
        if ((s2t.count(x) && s2t[x] != y) || (t2s.count(y) && t2s[y] != x)) return false;
        s2t[x] = y;
        t2s[y] = x;
    }
    return true;
}
```

## 7. 回文子字符串个数

中等： [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```html
Input: "aaa"	Output: 6	Explanation: "a", "a", "a", "aa", "aa", "aaa".
```

中心拓展法：从字符串的某一位开始，尝试着去扩展子字符串。

```java
private int cnt = 0;
public int countSubstrings(String s) {
    for (int i = 0; i < s.length(); i++) {
        extendSubstrings(s, i, i);     // 奇数长度
        extendSubstrings(s, i, i + 1); // 偶数长度
    }
    return cnt; 
}
private void extendSubstrings(String s, int start, int end) {
    while (start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
        start--;
        end++;
        cnt++;
    }
}
```

## 8. 判断回文数

简单：[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

要求不能使用额外空间，也就不能将整数转换为字符串进行判断。思路是反转一半数字，将整数分成左右两部分，右边那部分需要转置，然后判断这两部分是否相等。时间复杂度是以10为底n的对数。

整个过程我们不断将原始数字除以 `10`，然后给反转后的数字乘上 `10`，所以，当原始数字小于或等于反转后的数字时，就意味着我们已经处理了一半位数的数字了。 

<img src="../../../ZJW-Summary/assets/1622532438616.png" alt="1622532438616" style="zoom:67%;" />

```java
public boolean isPalindrome(int x) {
    if (x == 0) return true;
    if (x < 0 || x % 10 == 0) return false; // 10、100、1000、
    //right做的事情是把一次次获取x右边的个位数，然后一次次成，拼成x的右半部分
    //x /= 10 每次都消去右边一个个位。最后只剩下，x的左半部分。
    int right = 0;
    while (x > right) {//12321>0  1232>1  123>12 12>123 退出			11>0  1>1 退出
        right = right * 10 + x % 10;//r=0+1=1 10+2=12 120+3=123		 r=0+1=1  
        x /= 10; //x往右边移动一位，抹去一位数。  x=1232  123  12			x=1
    }
    return x == right || x == right / 10;//12==123/10				 1==1
}
```

## 9. 计数二进制子串

简单： [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/)

参考：

https://leetcode-cn.com/problems/count-binary-substrings/solution/ji-shu-er-jin-zhi-zi-chuan-by-leetcode-solution/

```js
输入: "00110011"	输出: 6		输入: "10101"	输出: 4
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
重复出现的子串要计算它们出现的次数。另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
```

思路：将字符串 s 按照 0 和 1 的连续段分组，存在 counts 数组中，例如 s = 00111011，可以得到这样的 counts 数组： = \{2, 3, 1, 2\}。这个 s 应该输出 4。这里 counts 数组中两个相邻的数一定代表的是两种不同的字符。

假设 counts 数组中两个相邻的数字为 u 或者 v，它们对应着 u 个 0 和 v 个 1，或者 u 个 1 和 v 个 0。能组成的满足条件的子串数目为 min{u,v}，即一对相邻的数字对答案的贡献。遍历所有相邻的数对，求它们的贡献总和。

```java
public int countBinarySubstrings(String s) {
    List<Integer> counts = new ArrayList<Integer>();
    int ptr = 0, n = s.length();
    while (ptr < n) {
        char c = s.charAt(ptr);	//获取当前被计数位置的字符
        int count = 0;
        while (ptr < n && s.charAt(ptr) == c) {
            ++ptr;
            ++count;
        }
        counts.add(count);
    }
    int ans = 0;
    for (int i = 1; i < counts.size(); ++i) {
        ans += Math.min(counts.get(i), counts.get(i - 1));
    }
    return ans;
}
```

其实我们只关心 i - 1位置的 counts 值是多少，换上滚动变量后，时间O(N) ，空间O(1)

```java
public int countBinarySubstrings(String s) {
    int ptr = 0, n = s.length(), cur = 0, ans = 0;
    while (ptr < n) {
        char c = s.charAt(ptr);
        int count = 0;
        while (ptr < n && s.charAt(ptr) == c) {
            ++ptr;
            ++count;
        }
        ans += Math.min(count, cur);
        cur = count;
    }
    return ans;
}
```






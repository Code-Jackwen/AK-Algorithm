## 字符串编辑

### 0. 判断子序列

简单：[392. 判断子序列](https://leetcode-cn.com/problems/is-subsequence/)

```
输入：s = "abc", t = "ahbgdc"	输出：true
进阶：大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，依次检查它们是否为 T 的子序列。
```

```java
//库函数
public boolean isSubsequence(String s, String t) {
    int idx = -1;
    for (char c : s.toCharArray()) {
        idx = t.indexOf(c, index + 1);
        if (idx == -1) return false;
    }
    return true;
}
```

```c++
//双指针
bool isSubsequence(string s, string t) {
    int n = s.length(), m = t.length();
    int i = 0, j = 0;
    while (i < n && j < m) {
        if (s[i] == t[j]) i++;
        j++;
    }
    return i == n;
}
```

动态规划，字符串编辑的入门题目

<img src="../../../ZJW-Summary/assets/1621500019830.png" alt="1621500019830" style="zoom: 50%;" />

**dp\[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp\[i][j]**。

if (s[i - 1] == t[j - 1])	t中找到了一个字符在s中也出现了， 那么dp\[i][j] = dp\[i - 1][j - 1] + 1 

if (s[i - 1] != t[j - 1])	相当于t要删除元素，继续匹配，那么 dp\[i][j] = dp\[i][j - 1]; 

从递推公式可以看出dp\[i][j]都是依赖于dp\[i - 1][j - 1] 和 dp\[i][j - 1]，所以dp\[0][0]和dp\[i][0]是一定要初始化的。

这里dp\[i][0]和dp\[0][j]是没有含义的，仅仅是为了给递推公式做前期铺垫，所以初始化为0，我理解为 ""。

```c++
bool isSubsequence(string s, string t) {
    vector<vector<int>> dp(s.size() + 1, vector<int>(t.size() + 1, 0));
    for (int i = 1; i <= s.size(); i++) {
        for (int j = 1; j <= t.size(); j++) {
            if (s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = dp[i][j - 1];
        }
    }
    if (dp[s.size()][t.size()] == s.size()) return true;
    return false;
}
```

s[i - 1]是中 i-1 因为 c++ 字符串访问的原因。

### 1. 删除后判断子序列

困难：[583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

```js
给俩字符串，每次可以删除任意一个字符串中的一个字符。给的字符串长度不超过500。字符串字符只含有小写字母。

输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```

可以转换为求两个字符串的最长公共子序列问题。

```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            //if (i == 0 || j == 0) continue;
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {		//相等
                dp[i][j] = dp[i - 1][j - 1] + 1;					//延续左上角
            } else {
                dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);	//比较左和上
            }
        }
    }
    return m + n - 2 * dp[m][n];
}
```

### 2. 删除后子序列的个数

困难：[115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

一、状态含义：dp\[i][j]：以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp\[i][j]。 

二、公式推导：这一类问题，基本是要分析两种情况

​		1、当s[i - 1] 与 t[j - 1]不相等时，dp\[i][j]只有一部分组成，不用s[i - 1]来匹配.

​				即：dp\[i][j] = dp\[i - 1][j];

​		2、当s[i - 1] 与 t[j - 1]相等时，dp\[i][j]可以有两部分组成。

​				一部分是用s[i - 1]来匹配，那么个数为dp\[i - 1][j - 1]。

​				一部分是不用s[i - 1]来匹配，个数为dp\[i - 1][j]。

 为什么还要考虑 不用s[i - 1]来匹配 ？

例如： s：bagg 和 t：bag ，s[3] 和 t[2]是相同的，但是字符串s也可以不用s[3]来匹配，即用s[0]s[1]s[2]组成的bag。

当然也可以用s[3]来匹配，即：s[0]s[1]s[3]组成的bag。

所以当s[i - 1] 与 t[j - 1]相等时，dp[i][j] = dp\[i - 1][j - 1] + dp\[i - 1][j];

三、初始化

1、dp\[i][0] 表示：以i-1为结尾的s可以随便删除元素，出现空字符串的个数。

s删除所有元素就成空串了，也匹配了，那么dp\[i][0]一定都是1。

2、dp\[0][j]，dp\[0][j]：空字符串s可以随便删除元素，出现以j-1为结尾的字符串t的个数。

那么dp\[0][j]一定都是0，s本身就是空串，咋删除也变成不了t。

3、dp\[0][0]应该是1，空字符串s，可以删除0个元素，变成空字符串t。

四、遍历顺序，看推导公式，可以看出dp\[i][j]都是根据左上方和正上方推出来的，因此从左到右，从上到下。

<img src="../../../ZJW-Summary/assets/1621504975532.png" alt="1621504975532" style="zoom:50%;" />

```c++
int numDistinct(string s, string t) {
    vector<vector<uint64_t>> dp(s.size() + 1, vector<uint64_t>(t.size() + 1));
    for (int i = 0; i < s.size(); i++) dp[i][0] = 1;
    for (int j = 1; j < t.size(); j++) dp[0][j] = 0;
    for (int i = 1; i <= s.size(); i++) {
        for (int j = 1; j <= t.size(); j++) {
            if (s[i - 1] == t[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[s.size()][t.size()];
}
```

参考： [uint8_t / uint16_t / uint32_t /uint64_t 是什么数据类型](https://blog.csdn.net/kiddy19850221/article/details/6655066?utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-1.control) 

总结：用 int 的话有个长的用例会爆掉，typedef long long int64_t;	typedef unsigned long long uint64_t; 

### 3. 编辑距离

困难：[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```js
修改一个字符串成为另一个字符串，求最少修改次数。一次修改操作包括：插入一个字符、删除一个字符、替换一个字符。

输入：word1 = "intention", word2 = "execution" 	输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

提示：0 <= word1.length, word2.length <= 500,word1 和 word2 由小写英文字母组成
```

 [【编辑距离】入门动态规划，你定义的 dp 里到底存了啥 - 编辑距离 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/edit-distance/solution/edit-distance-by-ikaruga/) 

`dp[i][j]` 代表 `word1` 中前 `i` 个字符，变换到 `word2` 中前 `j` 个字符，最短需要操作的次数 

需要考虑 `word1` 或 `word2` 一个字母都没有，即全增加/删除的情况，所以预留 `dp[0][j]` 和 `dp[i][0]` 

增，dp\[i][j] = dp\[i][j - 1] + 1

删，dp\[i][j] = dp\[i - 1][j] + 1

改，dp\[i][j] = dp\[i - 1][j - 1] + 1

如果刚好两个字母相同 `word1[i - 1] = word2[j - 1]`， 可以直接参考 `dp[i - 1][j - 1]` ，操作不用加一

<img src="../../../ZJW-Summary/assets/1621516969080.png" alt="1621516969080" style="zoom:50%;" />

```java
public int minDistance(String word1, String word2) {
    if (word1 == null || word2 == null) return 0;
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) dp[i][0] = i;
    for (int i = 1; i <= n; i++) dp[0][i] = i;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i - 1][j - 1], 
                                    Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
            }
        }
    }
    return dp[m][n];
}
```

### 4. 复制粘贴的最少次数

中等：[650. 只有两个键的键盘](https://leetcode-cn.com/problems/2-keys-keyboard/)

最开始只有一个字符 A，问需要多少次操作能够得到 n 个字符 A，每次操作可以复制当前所有的字符，或者粘贴。

除却第一次复制粘贴一个 A 之外，后面可以分析出，如果可以复制粘贴，肯定复制粘贴新的 A 串会比粘贴上一个复制的 A 串效率要高。那么如何判断是否可以完成复制粘贴呢，可以简单分析出，（n - 当前 A 串长度）% (当前 A 串长度) == 0 则可以复制粘贴。

```c
int minSteps(int n) {
    if(n == 1) return 0;
    int count = 1 ,ant = 1,fuzhi = 1; //当前 A 串长度，复制粘贴总次数，复制串的长度 
    int flag = 1; 					  // 复制之后就得粘贴，否则会死循环
    while(n!=count){
        if((n-count)%count==0 && flag == 0){	//复制全部
            fuzhi = count; 
            flag = 1;
        }else {									//粘贴
            count +=fuzhi;
            flag = 0;
        }
        ant++;
    }		
    return ant; 								//测试用例：3、4、5、6
}
```

```java
public int minSteps(int n) {
    if (n == 1) return 0;
    for (int i = 2; i <= Math.sqrt(n); i++) {
        if (n % i == 0) return i + minSteps(n / i);
    }
    return n;
}
```

```java
public int minSteps(int n) {
    int[] dp = new int[n + 1];
    int h = (int) Math.sqrt(n);
    for (int i = 2; i <= n; i++) {
        dp[i] = i;
        for (int j = 2; j <= h; j++) {
            if (i % j == 0) {
                dp[i] = dp[j] + dp[i / j];
                break;
            }
        }
    }
    return dp[n];
}
```


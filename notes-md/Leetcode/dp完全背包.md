## 完全背包

[3. 完全背包问题 - AcWing题库](https://www.acwing.com/problem/content/3/) 

原始的「完全背包」：要求不超过容积为 amount 的背包，让我们求的是总价值最多。

有 N件物品和一个容量是 V 的背包。每件物品只能使用一次，每种物品都有无限件可用。第 i 件物品的体积是 w[i]
，价值是 v[i]。求将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。输出最大价值。

**分析**：「完全背包问题」的重点在于：每种物品都有无限件可用，且不计算顺序。

**状态**：`dp[i][j]` 表示考虑物品区间 `[0, i]` 里，不超过背包容量，能够获得的最大价值； 

```js
方程：dp[i][j] = max(dp[i−1][j] , dp[i−1][j− k × w[i]] + k × v[i])		这里 k>= 0
```

**复杂度分析**：

说明：这一版代码的时间复杂度很高，使用了三重循环，**有重复计算**。 

- 时间复杂度：O(NW^2)，这里 N 是背包价值数组的长度，W 是背包的容量；
- 空间复杂度：O(NW)。

```java
//原始版本
public static int[][] npack(int W, int N, int[] weights, int[] values) {
    int[][] dp = new int[N + 1][W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = 0; j <= W; j++) {//j从1开始也对
            for (int k = 0; k * w <= j; k++) {
                dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - k * w] + k * v);
            }
            //dp[i][j] = dp[i - 1][j];
            //if (j - w >= 0) dp[i][j] = Math.max(dp[i][j], dp[i][j - w] + v);
        }
    }
    return dp;
}
```

优化掉里面的for循环，原理是不断使用前面已经更新过的 f \[ i ][ 前面体积 ]来更新 f \[ i ][ 后面的体积 j ] 

```java
//优化掉里面的for循环
public static int[][] npack(int W, int N, int[] weights, int[] values) {
    int[][] dp = new int[N + 1][W + 1];
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = 0; j <= W; j++) {
            dp[i][j] = dp[i - 1][j];
            if (j - w >= 0) dp[i][j] = Math.max(dp[i][j], dp[i][j - w] + v);
        }
    }
    return dp;
}
```

对比01背包

```java
public int[][] knapsack(int W, int N, int[] weights, int[] values) {
    int[][] dp = new int[N + 1][W + 1];
    for (int i = 1; i <= N; i++) {				
        int w = weights[i - 1], v = values[i - 1];
        for (int j = 1; j <= W; j++) {	
            dp[i][j] = dp[i - 1][j];		
            if (j >= w) dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - w] + v);
        }
    }	 
    return dp;
}
```

核心代码只有一行不同，对比：

```java
未去掉二维时，前提都有：dp[i][j] = dp[i - 1][j];

f[i][j] = max(f[i][j],f[i-1][j-w]+v);		//未去掉二维的01背包
f[i][j] = max(f[i][j],  f[i][j-w]+v);		//未去掉二维的完全背包问题

for (int j = W; j >= 1; j--) {				//去掉二维的01背包
    if (j >= w) dp[j] = Math.max(dp[j], dp[j - w] + v);		
}

for (int j = 1; j <= W; j++) {				//去掉二维的完全背包问题
    if (j >= w) dp[j] = Math.max(dp[j], dp[j - w] + v);		
}

if缩到for循环里，进而只是，01背包反序遍历，完全背包正序遍历的区别。
    
for (int j = W; j >= w; j--) {						//反序遍历
    dp[j] = Math.max(dp[j], dp[j - w] + v);			//01背包问题
}
    
for (int j = w; j <= W; j++) {						//正序遍历
    dp[j] = Math.max(dp[j], dp[j - w] + v);			//完全背包问题
}	
```

### 完全背包，最终版本

```java
public static int[] npack(int W, int N, int[] weights, int[] values) {
    int[] dp = new int[W + 1];							//只保留列
    for (int i = 1; i <= N; i++) {
        int w = weights[i - 1], v = values[i - 1];
        for (int j = w; j <= W; j++) {					//从j从w开始，正序遍历
            dp[j] = Math.max(dp[j], dp[j - w] + v);		//完全背包问题
        }
    }
    return dp;
}
```

```java
//测试
public static void main(String[] args) {
    int N = 4, W = 5;
    int[] weights = {1, 2, 3, 4}, values = {2, 4, 4, 5};
    int[][] dp = npack(W, N, weights, values);
    print(dp);//10	[0, 2, 4, 6, 8, 10]
}
```

### 4. 找零钱的最少硬币数

322\. Coin Change (Medium)

[Leetcode](https://leetcode.com/problems/coin-change/description/) / [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```js
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。
如果没有任何一种硬币组合能组成总金额，返回 -1。你可以认为每种硬币的数量是无限的。

输入：coins = [1, 2, 5], amount = 11
输出：3 	解释：11 = 5 + 5 + 1

输入：coins = [2], amount = 3
输出：-1

输入：coins = [1], amount = 0
输出：0
```

题目描述：给一些面额的硬币，要求用这些硬币来组成给定面额的钱数，并且使得硬币数量最少。

物品：硬币，**硬币可以重复使用。** 

- 物品大小：面额
- 物品价值：数量

因为硬币可以重复使用，不考虑元素组合之间的顺序，因此这是一个**完全背包问题**。

**完全背包只需要将 0-1 背包的逆序遍历 dp 数组改为正序遍历即可。**

一系列题目（未看完）： [一套框架解决「背包问题」 - 零钱兑换 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change/solution/yi-tao-kuang-jia-jie-jue-bei-bao-wen-ti-h0y40/) 

完全背包问题：

（1）如果是完全背包，即数组中的元素可重复使用并且不考虑元素之间顺序，arrs 放在外循环（保证 arrs 按顺序），target在内循环。且内循环正序。

（2）如果组合问题需考虑元素之间的顺序，需将 target 放在外循环，将 arrs 放在内循环。

本题转化为是否可以用 coins 中的数组合和成 amount，完全背包问题，并且为“不考虑排列顺序的完全背包问题”，外层循环为选择池 coins，内层循环为 amount。

对于元素之和等于 i - coin 的每一种组合，在最后添加 coin 之后即可得到一个元素之和等于 i 的组合，因此在计算 dp[i] 时，应该计算所有的 dp[i − coin] + 1 中的最小值。

dp[i] = min(dp[i], dp[i - coin] + 1)

dp[i] 表示和为 i 的 coin 组合中硬币最少有 dp[i] 个。

对于边界条件，我们定义 dp[0] = 0。最后返回 dp[amount] 

参考： [「代码随想录」322. 零钱兑换【完全背包】详解 - 零钱兑换 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change/solution/322-ling-qian-dui-huan-wan-quan-bei-bao-yged0/) 

```java
//21ms,28%
public int coinChange(int[] coins, int sum) {
    if (sum == 0 || coins == null) return 0;
    int[] dp = new int[sum + 1];
    Arrays.fill(dp, Integer.MAX_VALUE - 1);//如果初始为Integer.MAX_VALUE那么它+1就是
    dp[0] = 0;							   //-2147483648 解决方法可以是int类型的dp改为long
    for (int coin : coins) {			   //Arrays.fill(dp, sum + 1);
        for (int i = coin; i <= sum; i++) { //将逆序遍历改为正序遍历
            if (coin <= i)
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
        }
    }
    return dp[sum] == Integer.MAX_VALUE - 1 ? -1 : dp[sum];
}
```

```java
//其他，20ms,31%
public int coinChange(int[] coins, int sum) {
    if (sum == 0 || coins == null) return 0;
    int[] dp = new int[sum + 1];
    for (int coin : coins) {
        for (int i = coin; i <= sum; i++) { //将逆序遍历改为正序遍历
            if (i == coin) {
                dp[i] = 1;
            } else if (dp[i] == 0 && dp[i - coin] != 0) {
                dp[i] = dp[i - coin] + 1;
            } else if (dp[i - coin] != 0) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[sum] == 0 ? -1 : dp[sum];
}
```

### 5. 找零钱的硬币数组合

518\. Coin Change 2 (Medium)

[Leetcode](https://leetcode.com/problems/coin-change-2/description/) / [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

```js
给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的#硬币组合数。
假设每一种面额的#硬币有无限个。 

示例 1:
输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

示例 2:
输入: amount = 3, coins = [2]
输出: 0
解释: 只用面额2的硬币不能凑成总金额3。
示例 3:
输入: amount = 10, coins = [10] 
输出: 1

注意:
你可以假设：
0 <= amount (总金额) <= 5000、硬币种类不超过 500 种
1 <= coin (硬币面额) <= 5000、结果符合 32 位符号整数
```

 [零钱兑换 II（动态规划：完全背包问题 二维 / 一维☀） - 零钱兑换 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/coin-change-2/solution/ling-qian-dui-huan-iidong-tai-gui-hua-wa-tgrm/) 

方法一：二维（组合），由于每一种面额的金币可以选择多次，所以是完全背包问题。

**dp\[i][j]：表示背包容量为 j 时，在前 i 枚硬币中能组成的硬币组合个数。**

```java
public int change(int amount, int[] coins) {
    int n = coins.length;
    //i为第i个物品   j为容量
    int[][] dp = new int[n + 1][amount + 1];
    for (int i = 0; i <= n; i++) 	//第一列 amount 是0，所有的硬币都不放就是1中情况
        dp[i][0] = 1;
    for (int i = 1; i <= n; i++) {	//遍历硬币
        int c = coins[i - 1];
        for (int j = 1; j <= amount; j++) {//从小到大遍历目标值,目标值就是 背包大小
            if (j - c>= 0) {			   //放得下这个硬币
                dp[i][j] = dp[i - 1][j] + dp[i][j - c];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[n][amount];
}
```

完全背包问题，使用 dp 记录可达成目标的组合数目。套用完全背包模板：即数组中的元素可重复使用，coins 数组放在外循环，amount（背包容量）在内循环。且内循环正序。

dp[j]：表示当前背包容量为 j 时，可以凑成 j 的硬币组合数。

初始化：dp[0] = 1，当背包容量为 0 时，只有一种情况满足条件，就是一个硬币也不选。

```java
class Solution {
    public int change(int amount, int[] coins) {
        if (coins == null) return 0;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }
}
```

### 6. 字符串按单词列表分割

139\. Word Break (Medium)

[Leetcode](https://leetcode.com/problems/word-break/description/) / [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

```js
给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：#字典中没有重复的单词
拆分时#可以重复使用字典中的单词。

示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
```

dict 中的单词没有使用次数的限制，因此这是一个完全背包问题。

该问题涉及到字典中单词的使用顺序，也就是说物品必须按一定顺序放入背包中，例如下面的 dict 就不够组成字符串 "leetcode"：

```html
["lee", "tc", "cod"]
```

**求解顺序的完全背包问题时，对物品的迭代应该放在最里层，对背包的迭代放在外层，只有这样才能让物品按一定顺序放入背包中。**

<img src="../../../ZJW-Summary/assets/1621335242057.png" alt="1621335242057" style="zoom:80%;" />

```java
public boolean wordBreak(String s, List<String> wordDict) {
    int n = s.length();					 //背包
    boolean[] dp = new boolean[n + 1];
    dp[0] = true;
    for (int i = 1; i <= n; i++) {
        for (String word : wordDict) {   //对物品的迭代应该放在最里层
            int len = word.length();
            if (len <= i && word.equals(s.substring(i - len, i))) {
                dp[i] = dp[i] || dp[i - len];
            }//自己写字符串判断函数会快一些
        }
    }
    return dp[n];
}
```

 其他解法：[「手画图解」剖析三种解法: DFS, BFS, 动态规划 |139.单词拆分 - 单词拆分 ](https://leetcode-cn.com/problems/word-break/solution/shou-hui-tu-jie-san-chong-fang-fa-dfs-bfs-dong-tai/) 

 [字典树+DFS实现，击败90% - 单词拆分 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/word-break/solution/zi-dian-shu-dfsshi-xian-ji-bai-90-by-kua-48j6/) 

### 7. 组合总和

377\. Combination Sum IV (Medium)

[Leetcode](https://leetcode.com/problems/combination-sum-iv/description/) / [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

```js
给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。
请你从 nums 中找出并返回总和为 target 的元素组合的个数。
题目数据保证答案符合 32 位整数范围。

示例 1：
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
#(1, 1, 2)
(1, 2, 1)
(1, 3)
#(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。

示例 2：
输入：nums = [9], target = 3
输出：0
 
提示：
1 <= nums.length <= 200
1 <= nums[i] <= 1000
nums 中的所有元素 互不相同
1 <= target <= 1000
 
进阶：如果给定的数组允许负数出现，需要向题目中添加哪些限制条件？
```

本题是涉及顺序的完全背包，实际并不是组合，而是排列。对比在「完全背包」问题中，凑成总价值为 6 的方案 [1,2,3] 算是 1 种方案，但在本题算中是 3 * 2 * 1 = 63∗2∗1=6 种方案（[1,2,3] , [2,1,3] , [3,1,2] ... ）。

题目中 target 的范围是最小值是1，而 f[0] = 1; 没有任何实际意义，为的是 f[i] += f[i - u];  仅是为推导递推公式。 

**如果求组合数就是外层for循环遍历物品，内层for遍历背包**。

**如果求排列数就是外层for遍历背包，内层for循环遍历物品**。对比： [动态规划：518.零钱兑换II](https://mp.weixin.qq.com/s/PlowDsI4WMBOzf3q80AksQ)  是求的组合

如果把遍历nums（物品）放在外循环，遍历target的作为内循环的话，举一个例子：计算dp[4]的时候，结果集只有 {1,3} 这样的集合，不会有{3,1}这样的集合，因为nums遍历放在外层，3只能出现在1后面！

所以本题遍历顺序最终遍历顺序：target（背包）放在外循环，将nums（物品）放在内循环，内循环从前到后遍历。

 [「代码随想录」377. 组合总和 Ⅳ【动态规划】详解 - 组合总和 Ⅳ - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/combination-sum-iv/solution/dai-ma-sui-xiang-lu-377-zu-he-zong-he-iv-pj9s/) 

```java
public int combinationSum4(int[] nums, int tar) {
    int[] f = new int[tar + 1];
    f[0] = 1;
    for (int i = 1; i <= tar; i++) {
        for (int u : nums) {
            if (i >= u) f[i] += f[i - u];
        }
    }
    return f[tar];
}
```

```java
public int combinationSum4(int[] nums, int target) {
    if (nums == null || nums.length == 0) return 0;
    int[] maximum = new int[target + 1];
    maximum[0] = 1;
    Arrays.sort(nums);
    for (int i = 1; i <= target; i++) {
        for (int j = 0; j < nums.length && nums[j] <= i; j++) {
            maximum[i] += maximum[i - nums[j]];
        }
    }
    return maximum[target];
}
```


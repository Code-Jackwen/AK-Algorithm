# Leetcode 题解 - 搜索
<!-- GFM-TOC -->
* [Leetcode 题解 - 搜索](#leetcode-题解---搜索)
    * [BFS](#bfs)
        * [1. 计算在网格中从原点到特定点的最短路径长度](#1-计算在网格中从原点到特定点的最短路径长度)
        * [2. 组成整数的最小平方数数量](#2-组成整数的最小平方数数量)
        * [3. 最短单词路径](#3-最短单词路径)
    * [DFS](#dfs)
        * [1. 查找最大的连通面积](#1-查找最大的连通面积)
        * [2. 矩阵中的连通分量数目](#2-矩阵中的连通分量数目)
        * [3. 好友关系的连通分量数目](#3-好友关系的连通分量数目)
        * [4. 填充封闭区域](#4-填充封闭区域)
        * [5. 能到达的太平洋和大西洋的区域](#5-能到达的太平洋和大西洋的区域)
    * [Backtracking](#backtracking)
        * [1. 数字键盘组合](#1-数字键盘组合)
        * [2. IP 地址划分](#2-ip-地址划分)
        * [3. 在矩阵中寻找字符串](#3-在矩阵中寻找字符串)
        * [4. 输出二叉树中所有从根到叶子的路径](#4-输出二叉树中所有从根到叶子的路径)
        * [5. 排列](#5-排列)
        * [6. 含有相同元素求排列](#6-含有相同元素求排列)
        * [7. 组合](#7-组合)
        * [8. 组合求和](#8-组合求和)
        * [9. 含有相同元素的组合求和](#9-含有相同元素的组合求和)
        * [10. 1-9 数字的组合求和](#10-1-9-数字的组合求和)
        * [11. 子集](#11-子集)
        * [12. 含有相同元素求子集](#12-含有相同元素求子集)
        * [13. 分割字符串使得每个部分都是回文数](#13-分割字符串使得每个部分都是回文数)
        * [14. 数独](#14-数独)
        * [15. N 皇后](#15-n-皇后)
<!-- GFM-TOC -->


深度优先搜索和广度优先搜索广泛运用于树和图中，但是它们的应用远远不止如此。

## BFS

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/95903878-725b-4ed9-bded-bc4aae0792a9.jpg"/> </div><br>
广度优先搜索一层一层地进行遍历，每层遍历都是以上一层遍历的结果作为起点，遍历一个距离能访问到的所有节点。需要注意的是，遍历过的节点不能再次被遍历。

第一层：

- 0 -\> {6,2,1,5}

第二层：

- 6 -\> {4}
- 2 -\> {}
- 1 -\> {}
- 5 -\> {3}

第三层：

- 4 -\> {}
- 3 -\> {}

每一层遍历的节点都与根节点距离相同。设 d<sub>i</sub> 表示第 i 个节点与根节点的距离，推导出一个结论：对于先遍历的节点 i 与后遍历的节点 j，有 d<sub>i</sub> <= d<sub>j</sub>。利用这个结论，可以求解最短路径等   **最优解**   问题：第一次遍历到目的节点，其所经过的路径为最短路径。应该注意的是，使用 **BFS 只能求解无权图的最短路径，无权图是指从一个节点到另一个节点的代价都记为 1。**

在程序实现 BFS 时需要考虑以下问题：

- 队列：用来存储每一轮遍历得到的节点；
- 标记：对于遍历过的节点，应该将它标记，防止重复遍历。

### 1. 计算在网格中从原点到特定点的最短路径长度

1091\. Shortest Path in Binary Matrix(Medium)

[Leetcode](https://leetcode.com/problems/shortest-path-in-binary-matrix/) / [力扣](https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/)

```html
给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。

二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：

1、路径途经的所有单元格都的值都是 0 。
2、路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
3、畅通路径的长度 是该路径途经的单元格总数。

示例 3：
输入：grid = [[1,0,0],[1,1,0],[1,1,0]]
输出：-1

提示：
n == grid.length
n == grid[i].length
1 <= n <= 100
grid[i][j] 为 0 或 1
```

```js
示例 1：
输入：grid = [[0,1],[1,0]]
输出：2
```

<img src="../../assets/1618125364615.png" alt="1618125364615" style="zoom: 50%;" />

```js
示例 2：
输入：grid = [[0,0,0],[1,1,0],[1,1,0]]
输出：4
```

<img src="../../assets/1618125417485.png" alt="1618125417485" style="zoom:50%;" />

题目描述：0 表示可以经过某个位置，求解从左上角到右下角的最短路径长度。

```java
class Solution {
    public int shortestPathBinaryMatrix(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        if(grid[0][0]==1 || grid[grid.length-1][grid[0].length-1]==1) return -1;
        grid[0][0] = 1;
        Queue<int[]> q=new LinkedList<>();
        q.add(new int[]{0, 0});
        int len = q.size();										// 队列长度
        int c = 0;												// c要和队列长度比
        int[][] dir = {{1, 0}, {1, 1}, {1,-1}, {0, 1}, 			// 定义八个方向
                       {0, -1}, {-1, 0},{-1, -1}, {-1, 1}};
        int path = 1;											// 定义最短路径长度
        while(!q.isEmpty()){
            int[] data = q.poll();
            int x = data[0];									// x、y代表当前的坐标（x，y）
            int y = data[1];
            if(x == m - 1 && y == n - 1)  return path;			// 先到先返回
            for(int[] d : dir){
                int x1 = x + d[0];								// 移动后的坐标（x1，y1）
                int y1 = y + d[1];
                if(x1 >= 0 && y1 >= 0 && x1 < m && y1 < n && grid[x1][y1] == 0){
                    q.add(new int[]{x1, y1});					// 将可走的步添加进队列
                    grid[x1][y1] = 1;							//标记
                }
            }
            c++;												//计算步数的核心
            if(c == len){	//判断的逻辑是把上次队列存储全部节点处理完毕后，path++
                c = 0;		//len等于这层节点的数量
                path++;
                len = q.size();
            }
        }
        return -1;
    }
}
```

下边这个也是上边的思路好理解一些

<img src="../../assets/1618135273853.png" alt="1618135273853" style="zoom:50%;" />

```java
class Solution {
     public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        if (grid[0][0] == 1 || grid[n - 1][m - 1] == 1) return -1;
        if (n == 1 && m == 1) return 1;
        int[] dx = new int[]{-1, -1, -1,  0, 0, 1, 1, 1};
        int[] dy = new int[]{-1,  0,  1, -1, 1,-1, 0, 1};
        grid[0][0] = 1;
        Queue<int[]> queue = new LinkedList<>();				//一维数组节点
        queue.add(new int[]{0, 0});
        int count = 0;
        while (!queue.isEmpty()) {
            count++;
            int queueSize = queue.size();
            for (int i = 0; i < queueSize; i++) {				//一层的节点数处理完后
                int[] data = queue.poll();						//count++;
                if (data[0] == n - 1 && data[1] == m - 1) return count;
                for (int j = 0; j < 8; j++) {
                    int x = data[0] + dx[j];
                    int y = data[1] + dy[j];
                    if (x >= 0 && x < n && y >= 0 && y < m && grid[x][y] == 0 ) {
                        queue.add(new int[]{x, y});
                        grid[x][y] = 1;
                    }
                }
            }
        }
        return -1;
    }
}
```

其他

```java
public int shortestPathBinaryMatrix(int[][] grids) {
        if (grids == null || grids.length == 0 || grids[0].length == 0) {
            return -1;
        }
        int[][] direction = {{1, -1}, {1, 0}, {1, 1}, {0, -1}, {0, 1}, {-1, -1}, {-1, 0}, {-1, 1}};
        int m = grids.length, n = grids[0].length;
        Queue<Pair<Integer, Integer>> queue = new LinkedList<>();
        queue.add(new Pair<>(0, 0));
        int pathLength = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            pathLength++;
            while (size-- > 0) {
                Pair<Integer, Integer> cur = queue.poll();
                int cr = cur.getKey(), cc = cur.getValue();
                if (grids[cr][cc] == 1) {
                    continue;
                }
                if (cr == m - 1 && cc == n - 1) {
                    return pathLength;
                }
                grids[cr][cc] = 1; // 标记
                for (int[] d : direction) {
                    int nr = cr + d[0], nc = cc + d[1];
                    if (nr < 0 || nr >= m || nc < 0 || nc >= n) {
                        continue;
                    }
                    queue.add(new Pair<>(nr, nc));
                }
            }
        }
        return -1;
    }
```

### 2. 组成整数的最小平方数数量

279\. Perfect Squares (Medium)

[Leetcode](https://leetcode.com/problems/perfect-squares/description/) / [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```js
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。
你需要让组成和的完全平方数的个数最少。
给你一个整数 n ，返回#和为 n 的完全平方数的 最少数量 。
完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

示例 1：
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4

示例 2：
输入：n = 13
输出：2
解释：13 = 4 + 9
 
提示：1 <= n <= 104
```

可以将每个整数看成**图**中的一个节点，如果两个整数之差为一个平方数，那么这两个整数所在的节点就有一条边。

要求解最小的平方数数量，就是求解从节点 n 到节点 0 的最短路径。

本题也可以用动态规划求解，在之后动态规划部分中会再次出现。

```java
public int numSquares(int n) {
    List<Integer> squares = generateSquares(n);
    Queue<Integer> queue = new LinkedList<>();
    boolean[] marked = new boolean[n + 1];
    queue.add(n);
    marked[n] = true;
    int level = 0;
    while (!queue.isEmpty()) {
        int size = queue.size();
        level++;
        while (size-- > 0) {
            int cur = queue.poll();
            for (int s : squares) {
                int next = cur - s;
                if (next < 0) {
                    break;
                }
                if (next == 0) {
                    return level;
                }
                if (marked[next]) {
                    continue;
                }
                marked[next] = true;
                queue.add(next);
            }
        }
    }
    return n;
}

/**
 * 生成小于 n 的平方数序列
 * @return 1,4,9,16...
 */
private List<Integer> generateSquares(int n) {
    List<Integer> squares = new ArrayList<>();
    int square = 1;
    int diff = 3;
    while (square <= n) {
        squares.add(square);
        square += diff;
        diff += 2;
    }
    return squares;
}
```

### 3. 最短单词路径

127\. Word Ladder (Medium)

[Leetcode](https://leetcode.com/problems/word-ladder/description/) / [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

```js
字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：
1、序列中第一个单词是 beginWord 。
2、序列中最后一个单词是 endWord 。
3、每次转换只能改变一个字母。
4、转换过程中的中间单词必须是字典 wordList 中的单词。
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。
 
示例 1：
输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
输出：5
解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
i编程o，h变成d，t变成g，d变成c

示例 2：
输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
输出：0
解释：endWord "cog" 不在字典中，所以无法进行转换。

提示：
1 <= beginWord.length <= 10
endWord.length == beginWord.length
1 <= wordList.length <= 5000
wordList[i].length == beginWord.length
beginWord、endWord 和 wordList[i] 由小写英文字母组成
beginWord != endWord
wordList 中的所有字符串 互不相同
```

题目描述：找出一条从 beginWord 到 endWord 的最短路径，每次移动规定为改变一个字符，并且改变之后的字符串必须在 wordList 中。

```js

```

题解

- 无向图中两个顶点之间的最短路径的长度，可以通过广度优先遍历得到；
- 为什么 BFS 得到的路径最短？可以把起点和终点所在的路径拉直来看，两点之间线段最短；
- 已知目标顶点的情况下，可以分别从起点和目标顶点（终点）执行广度优先遍历，直到遍历的部分有交集，是双向广度优先遍历的思想。

### todo

参考：BFS、双向BFS

https://leetcode-cn.com/problems/word-ladder/solution/yan-du-you-xian-bian-li-shuang-xiang-yan-du-you-2/



其他，这个1000+ms，太慢了

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    wordList.add(beginWord);
    int N = wordList.size();
    int start = N - 1;
    int end = 0;
    while (end < N && !wordList.get(end).equals(endWord)) {
        end++;
    }
    if (end == N) {
        return 0;
    }
    List<Integer>[] graphic = buildGraphic(wordList);
    return getShortestPath(graphic, start, end);
}

private List<Integer>[] buildGraphic(List<String> wordList) {
    int N = wordList.size();
    List<Integer>[] graphic = new List[N];
    for (int i = 0; i < N; i++) {
        graphic[i] = new ArrayList<>();
        for (int j = 0; j < N; j++) {
            if (isConnect(wordList.get(i), wordList.get(j))) {
                graphic[i].add(j);
            }
        }
    }
    return graphic;
}

private boolean isConnect(String s1, String s2) {
    int diffCnt = 0;
    for (int i = 0; i < s1.length() && diffCnt <= 1; i++) {
        if (s1.charAt(i) != s2.charAt(i)) {
            diffCnt++;
        }
    }
    return diffCnt == 1;
}

private int getShortestPath(List<Integer>[] graphic, int start, int end) {
    Queue<Integer> queue = new LinkedList<>();
    boolean[] marked = new boolean[graphic.length];
    queue.add(start);
    marked[start] = true;
    int path = 1;
    while (!queue.isEmpty()) {
        int size = queue.size();
        path++;
        while (size-- > 0) {
            int cur = queue.poll();
            for (int next : graphic[cur]) {
                if (next == end) {
                    return path;
                }
                if (marked[next]) {
                    continue;
                }
                marked[next] = true;
                queue.add(next);
            }
        }
    }
    return 0;
}
```

## DFS

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/74dc31eb-6baa-47ea-ab1c-d27a0ca35093.png"/> </div><br>
广度优先搜索一层一层遍历，每一层得到的所有新节点，要用队列存储起来以备下一层遍历的时候再遍历。

而深度优先搜索在得到一个新节点时立即对新节点进行遍历：从节点 0 出发开始遍历，得到到新节点 6 时，立马对新节点 6 进行遍历，得到新节点 4；如此反复以这种方式遍历新节点，直到没有新节点了，此时返回。返回到根节点 0 的情况是，继续对根节点 0 进行遍历，得到新节点 2，然后继续以上步骤。

从一个节点出发，使用 DFS 对一个图进行遍历时，能够遍历到的节点都是从初始节点可达的，DFS 常用来求解这种   **可达性**   问题。

在程序实现 DFS 时需要考虑以下问题：

- 栈：用栈来保存当前节点信息，当遍历新节点返回时能够继续遍历当前节点。可以使用递归栈。
- 标记：和 BFS 一样同样需要对已经遍历过的节点进行标记。

### 1. 查找最大的连通面积

695\. Max Area of Island (Medium)

[Leetcode](https://leetcode.com/problems/max-area-of-island/description/) / [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```js
给定一个包含了一些 0 和 1 的非空二维数组 grid 。
一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，#「相邻」要求两个 1 必须在水平或者竖直方向上相邻。
你可以假设 grid 的四个边缘都被 0（代表水）包围着。
找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

示例 1:
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,#1,0,1,0,0],
 [0,1,0,0,1,1,0,0,#1,1,1,0,0],
 [0,0,0,0,0,0,0,0,#0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。

示例：	返回4
[[1,1,0,0,0],
 [1,1,0,0,0],
 [0,0,0,1,1],
 [0,0,0,1,1]]
求的是各个岛屿中最大的一个面积，而不是岛屿个数。

注意: 给定的矩阵grid 的长度和宽度都不超过 50。
```

标准DFS，这个更快。

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(res, dfs(i, j, grid));
                }
            }
        }
        return res;
    }
    // 每次调用的时候默认num为1，进入后判断如果不是岛屿，则直接返回0，就可以避免预防错误的情况。
    // 每次找到岛屿，则直接把找到的岛屿改成0，这是传说中的沉岛思想，就是遇到岛屿就把他和周围的全部沉默。
    // ps：如果能用沉岛思想，那么自然可以用朋友圈思想。有兴趣的朋友可以去尝试。
    private int dfs(int i, int j, int[][] grid) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[i].length
         || grid[i][j] == 0) {	return 0;  }
        
        grid[i][j] = 0;
        int s = 1;
        s += dfs(i + 1, j, grid);
        s += dfs(i - 1, j, grid);
        s += dfs(i, j + 1, grid);
        s += dfs(i, j - 1, grid);
        return s;

    }
}
```

其他

```java
class Solution {
    private int m, n;
    private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int maxAreaOfIsland(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        m = grid.length;
        n = grid[0].length;
        int maxArea = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxArea = Math.max(maxArea, dfs(grid, i, j));
            }
        }
        return maxArea;
    }

    private int dfs(int[][] grid, int r, int c) {
        if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0) {
            return 0;
        }
        grid[r][c] = 0;
        int area = 1;
        for (int[] d : direction) {
            area += dfs(grid, r + d[0], c + d[1]);
        }
        return area;
    }
}
```

### 2. 矩阵中的连通分量数目

200\. Number of Islands (Medium)

[Leetcode](https://leetcode.com/problems/number-of-islands/description/) / [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```html
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

示例 2：
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

提示：
m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] 的值为 '0' 或 '1'
```

```java
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[0].length; j++) {
                if(grid[i][j] == '1'){
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }
    private void dfs(char[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length 
           		 || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i - 1, j);
        dfs(grid, i, j - 1);
    }
}
```

其他，可以将矩阵表示看成一张有向图。

```java
//思路其实一样
class Solution {
    private int m, n;
    private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        m = grid.length;
        n = grid[0].length;
        int islandsNum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != '0') {
                    dfs(grid, i, j);
                    islandsNum++;
                }
            }
        }
        return islandsNum;
    }

    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        for (int[] d : direction) {
            dfs(grid, i + d[0], j + d[1]);
        }
    }
}
```

### 3. 好友关系的连通分量数目

547\. Friend Circles (Medium)

[Leetcode](https://leetcode.com/problems/friend-circles/description/) / [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

```js
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
返回矩阵中 省份 的数量。

提示：
1 <= n <= 200
n == isConnected.length、n == isConnected[i].length、矩阵都是正方形
isConnected[i][j] 为 1 或 0
isConnected[i][i] == 1
isConnected[i][j] == isConnected[j][i]
```

示例 1：

<img src="../../assets/1618193276869.png" alt="1618193276869" style="zoom:50%;" />

```
输入：
isConnected = 
[[1,1,0], 
 [1,1,0],
 [0,0,1]]
输出：2
```

示例 2：

<img src="../../assets/1618193254926.png" alt="1618193254926" style="zoom:50%;" />

```
输入：
isConnected = 
[[1,0,0],
 [0,1,0],
 [0,0,1]]
输出：3

输入：
{ {1,1,1},
  {1,1,1},
  {1,1,1} };
输出：1     

输入：
{ {0,0,0},
  {0,0,0},
  {0,0,0} };
输出：3
```

题目描述：好友关系可看成是一个无向图，例第 0 个人与第 1 个人是好友，那么 M\[0][1] 和 M\[1][0] 的值都为 1。

应该找一找图解的，没找到DFS的图解。

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        // int[][] isConnected 是无向图的邻接矩阵，n 为无向图的顶点数量
        int n = isConnected.length;
        boolean[] visited = new boolean[n];					//标识的是行，是否访问过。
        int cnt = 0;										//累计遍历过的连通域的数量
        for (int i = 0; i < n; i++) {
            // 若当前顶点 i 未被访问，说明又是一个新的连通域，则遍历新的连通域且cnt+=1.
            if (!visited[i]) { 
                cnt++;
                dfs(i, isConnected, visited);
            }
        }
        return cnt;
    }

    private void dfs(int i, int[][] isConnected, boolean[] visited) {
        visited[i] = true;									// 对当前顶点 i 进行访问标记
        for (int j = 0; j < isConnected.length; j++) {		// 继续遍历与顶点 i 相邻的顶点
            if (isConnected[i][j] == 1 && !visited[j]) {	//这里visited是j
                dfs(j, isConnected, visited);				//这里访问的是j
            }
        }
    }
}
```

BFS、并查集（解法比较典型），解法参考：https://leetcode-cn.com/problems/number-of-provinces/solution/dfs-bfs-bing-cha-ji-3-chong-fang-fa-ji-s-edkl/

### 4. 填充封闭区域

130\. Surrounded Regions (Medium)

[Leetcode](https://leetcode.com/problems/surrounded-regions/description/) / [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

```js
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

示例 1：
输入：board = 
    [["X","X","X","X"],
     ["X","O","O","X"],
     ["X","X","O","X"],
     ["X","O","X","X"]]

输出：[["X","X","X","X"],
      ["X","X","X","X"],
      ["X","X","X","X"],
      ["X","O","X","X"]]

解释：
被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 
任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。
如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

提示：
m == board.length
n == board[i].length
1 <= m, n <= 200
board[i][j] 为 'X' 或 'O'
```

题目描述：使被 'X' 包围的 'O' 转换为 'X'。思路，先填充最外侧，剩下的就是里侧了。

本题给定的矩阵中有三种元素：

- 字母 `X`；
- 被字母 `X` 包围的字母 `O`；
- 没有被字母 `X` 包围的字母 `O`。

本题要求将所有被字母 X 包围的字母 O都变为字母 X ，但很难判断哪些 O 是被包围的，哪些 O 不是被包围的。

注意到题目解释中提到：任何边界上的 O 都不会被填充为 X。 

我们可以想到，所有的不被包围的 O 都直接或间接与边界上的 O 相连。我们可以利用这个性质判断 O 是否在边界上，具体地说：

- 对于每一个边界上的 O，我们以它为起点，标记所有与它直接或间接相连的字母 O；
- 最后我们遍历这个矩阵，对于每一个字母：
  - 如果该字母被标记过，则该字母为没有被字母 X 包围的字母 O，我们将其还原为字母 O；
  - 如果该字母没有被标记过，则该字母为被字母 X 包围的字母 O，我们将其修改为字母 X。

思路及解法

先把和边界4个方向相连接的所有O 修改为字母 `T`，然后全局遍历，等于T的变成O，等于O的变成X。

其他：DFS、并查集，参考：https://leetcode-cn.com/problems/surrounded-regions/solution/bfsdi-gui-dfsfei-di-gui-dfsbing-cha-ji-by-ac_pipe/

```java
class Solution {
    int r, c;										//全局的行、列
    public void solve(char[][] board) {
        r = board.length;
        if (r == 0) return;
        c = board[0].length;
        for (int i = 0; i < r; i++) {				//最左和最右列
            dfs(board, i, 0);
            dfs(board, i, c - 1);
        }
        for (int j = 1; j < c - 1; j++) {			//第一行和最后一行，分别第二列到倒数第二列。
            dfs(board, 0, j);
            dfs(board, r - 1, j);
        }
        for (int i = 0; i < r; i++) {				//全部遍历
            for (int j = 0; j < c; j++) {
                if (board[i][j] == 'T') {			//等于T的变成O
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {	//等于O的变成X
                    board[i][j] = 'X';
                }
            }
        }
    }

    public void dfs(char[][] board, int x, int y) {
        if (x < 0 || x >= r || y < 0 || y >= c || board[x][y] != 'O') {
            return;
        }
        board[x][y] = 'T';							//常规4个方向的标记遍历
        dfs(board, x + 1, y);
        dfs(board, x - 1, y);
        dfs(board, x, y + 1);
        dfs(board, x, y - 1);
    }
}
```

### 5. 能到达的太平洋和大西洋的区域

417\. Pacific Atlantic Water Flow (Medium)

[Leetcode](https://leetcode.com/problems/pacific-atlantic-water-flow/description/) / [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

```js
给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。
规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。
请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

提示：输出坐标的顺序不重要、m 和 n 都小于150

示例：
给定下面的 5x5 矩阵:
  太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋
返回:
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
```

左边和上边是太平洋，右边和下边是大西洋，内部的数字代表海拔，海拔高的地方的水能够流到低的地方，求解水能够流到太平洋和大西洋的所有位置。

思路是从海洋开始逆流，如果可以逆流到，就标记为1，然后检查两个海洋都可以逆流到的区域。

BFS ，参考：

https://leetcode-cn.com/problems/pacific-atlantic-water-flow/solution/ni-liu-dfs-yu-bfs-by-fibonacciwh/

```java
class Solution {
    int r, c;							  //定义成全局的，减少dfs函数的形参
    int[][] matrix;
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return new ArrayList<>();
        }
        this.r = matrix.length;
        this.c = matrix[0].length;
        this.matrix = matrix;
        int[][] tp = new int[r][c];       //太平洋
        int[][] dx = new int[r][c];       //大西洋
        //从海洋边界开始
//        for (int i = 0; i < r; i++) {
//            for (int j = 0; j < c; j++) {
//                if (i == 0 || j == 0) {
//                    dfs( tp, i, j, matrix[i][j]);
//                }
//                if (i == r - 1 || j == c - 1) {
//                    dfs( dx, i, j, matrix[i][j]);
//                }
//            }
//        }

        for (int i = 0; i < r && i < c; i++) {
            dfs(tp, i, 0, matrix[i][0]);    //最左和最右边列
            dfs(dx, i, c - 1, matrix[i][c - 1]);

            dfs(tp, 0, i, matrix[0][i]);    //最上、最下
            dfs(dx, r - 1, i, matrix[r - 1][i]);
        }
        if (r > c) {           //处理剩余
            for (int i = c; i < r; i++) {
                dfs(tp, i, 0, matrix[i][0]);      //最左和最右边列
                dfs(dx, i, c - 1, matrix[i][c - 1]);
            }
        } else if (r < c) {
            for (int i = r; i < c; i++) {
                dfs(tp, 0, i, matrix[0][i]);            //最上、最下
                dfs(dx, r - 1, i, matrix[r - 1][i]);
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (tp[i][j] == 1 && dx[i][j] == 1) {
                    res.add(Arrays.asList(i, j));
                }
            }
        }
        return res;
    }
    private void dfs(int[][] aux, int i, int j, int pre) {
        if (i < 0 || j < 0 || i > r - 1 || j > c - 1
         || aux[i][j] == 1 || matrix[i][j] < pre) { 
            return;        // 流到过的，还有逆流的延伸的要比原来的大。
        }
        aux[i][j] = 1;                              //标记流到过了
        dfs(aux, i - 1, j, matrix[i][j]);
        dfs(aux, i + 1, j, matrix[i][j]);
        dfs(aux, i, j - 1, matrix[i][j]);
        dfs(aux, i, j + 1, matrix[i][j]);
    }
}
```

```java
class Solution {
    private List<List<String>> solutions;
    private char[][] nQueens;
    private boolean[] colUsed;
    private boolean[] diagonals45Used;
    private boolean[] diagonals135Used;
    private int n;

    public List<List<String>> solveNQueens(int n) {
        solutions = new ArrayList<>();
        nQueens = new char[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(nQueens[i], '.');
        }
        colUsed = new boolean[n];
        diagonals45Used = new boolean[2 * n - 1];
        diagonals135Used = new boolean[2 * n - 1];
        this.n = n;
        backtracking(0);
        return solutions;
    }

    private void backtracking(int row) {
        if (row == n) {
            List<String> list = new ArrayList<>();
            for (char[] chars : nQueens) {
                list.add(new String(chars));
            }
            solutions.add(list);
            return;
        }

        for (int col = 0; col < n; col++) {
            int diagonals45Idx = row + col;
            int diagonals135Idx = n - 1 - (row - col);
            if (colUsed[col] || diagonals45Used[diagonals45Idx] || diagonals135Used[diagonals135Idx]) {
                continue;
            }
            nQueens[row][col] = 'Q';
            colUsed[col] = diagonals45Used[diagonals45Idx] = diagonals135Used[diagonals135Idx] = true;
            backtracking(row + 1);
            colUsed[col] = diagonals45Used[diagonals45Idx] = diagonals135Used[diagonals135Idx] = false;
            nQueens[row][col] = '.';
        }
    }
}
```

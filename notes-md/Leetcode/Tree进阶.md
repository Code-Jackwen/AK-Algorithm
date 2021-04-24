# Leetcode 题解 - 树 - 进阶

* [BST](#bst)
  * [1. 修剪二叉查找树](#1-修剪二叉查找树)
  * [2. 寻找二叉查找树的第 k 个元素](#2-寻找二叉查找树的第-k-个元素)
  * [3. 把二叉查找树每个节点的值都加上比它大的节点的值](#3-把二叉查找树每个节点的值都加上比它大的节点的值)
  * [4. 二叉查找树的最近公共祖先](#4-二叉查找树的最近公共祖先)
  * [5. 二叉树的最近公共祖先](#5-二叉树的最近公共祖先)
  * [6. 从有序数组中构造二叉查找树](#6-从有序数组中构造二叉查找树)
  * [7. 根据有序链表构造平衡的二叉查找树](#7-根据有序链表构造平衡的二叉查找树)
  * [8. 在二叉查找树中寻找两个节点，使它们的和为一个给定值](#8-在二叉查找树中寻找两个节点，使它们的和为一个给定值)
  * [9. 在二叉查找树中查找两个节点之差的最小绝对值](#9-在二叉查找树中查找两个节点之差的最小绝对值)
  * [10. 寻找二叉查找树中出现次数最多的值](#10-寻找二叉查找树中出现次数最多的值)
* [Trie](#trie)
  * [1. 实现一个 Trie](#1-实现一个-trie)
  * [2. 实现一个 Trie，用来求前缀和](#2-实现一个-trie，用来求前缀和)

## BST

二叉查找树（BST）：根节点大于等于左子树**所有节点**，小于等于右子树**所有节点。**

**二叉查找树中序遍历有序。**

### 1. 修剪 - BST

669\. Trim a Binary Search Tree (Easy)

[Leetcode](https://leetcode.com/problems/trim-a-binary-search-tree/description/) / [669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/)

结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

<img src="../../../../../项目/Git-md/ZJW-Summary/assets/1619182989789.png" alt="1619182989789" style="zoom: 67%;" />

分界线

<img src="../../../../../项目/Git-md/ZJW-Summary/assets/1619183020408.png" alt="1619183020408" style="zoom:67%;" />

分界线

<img src="../../../ZJW-Summary/assets/1619186148844.png" alt="1619186148844" style="zoom:50%;" />

题目描述：输入BST 和 [low,high]，只保留值在 [low,high] 之间的节点

解题方法：迭代，非队列。改变头节点的指针。

[三种解法：递归，迭代，DFS - 修剪二叉搜索树 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/trim-a-binary-search-tree/solution/san-chong-jie-fa-di-gui-die-dai-dfs-by-r-ikyk/) 

```java
//0ms、100%
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) return root;
        // 找到修剪后的二叉搜索树的头节点root，该头节点root位于`[low,high]`区间内
        while (root != null && (root.val > high || root.val < low)) {//必须判空
            root = root.val > high ? root.left : root.right;
        }
        // 循环迭代处理root的左子树，将小于low的节点排除掉，剩下节点都在`[low,high]`区间内
        TreeNode cur = root;  //缓存下头
        while (cur != null) {  //操作头的左儿子
            while (cur.left != null && cur.left.val < low) {
                cur.left = cur.left.right;
            }
            cur = cur.left;
        }
        // 循环迭代处理root的右子树，将大于high的节点排除掉，剩下节点都在`[low,high]`区间内
        cur = root;     //指向头
        while (cur != null) {  //操作头的右儿子
            while (cur.right != null && cur.right.val > high) {
                cur.right = cur.right.left;
            }
            cur = cur.right;
        }
        return root;
    }
}
```

- 区间[low, high]位于root的左子树，则修剪后的二叉搜索树位于root的左子树。
- 区间[low, high]位于root的右子树，则修剪后的二叉搜索树位于root的右子树。
- 区间[low, high]包含root节点，则分别对root的左子树和右子树进行修剪，修剪完成后分别赋给root的左、右节点。

```java
//官解改编，为了透漏思路
public TreeNode trimBST(TreeNode root, int low, int high) {
    if (root == null) return null;
    if (root.val < low) {
        //因为是二叉搜索树,节点.left < 节点 < 节点.right
        //节点数字比low小,就把左节点全部裁掉.
        root = root.right;
        //裁掉之后,继续看右节点的剪裁情况.剪裁后重新赋值给root.
        root = trimBST(root, low, high);
    } else if (root.val > high) {
        //如果数字比high大,就把右节点全部裁掉.
        root = root.left;
        //裁掉之后,继续看左节点的剪裁情况
        root = trimBST(root, low, high);
    } else {
        //如果数字在区间内,就去裁剪左右子节点.
        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
    }
    return root;
}
```

```java
//官解
public TreeNode trimBST(TreeNode root, int l, int h) {
    if (root == null) return null;
    // 如果当前结点小于下界，直接将修剪后的右子树替换当前节点并返回
    if (root.val < l) return trimBST(root.right, l, h);
    // 如果当前结点大于上界，直接将修剪后的左子树替换当前节点并返回
    if (root.val > h) return trimBST(root.left, l, h);
    // 如果数字在区间内，继续往两边扩散寻找第一个越界的结点
    root.left = trimBST(root.left, l, h);
    root.right = trimBST(root.right, l, h);
    return root;
}
```

### 2. 第 k 小的元素 - BST

230\. Kth Smallest Element in a BST (Medium)

[Leetcode](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) / [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

设计一个算法查找其中第 `k` 个最小元素（从 1 开始计数）。 

中序遍历解法： 

```java
private int cnt = 0;
private int val;

public int kthSmallest(TreeNode root, int k) {
    inOrder(root, k);
    return val;
}

private void inOrder(TreeNode node, int k) {
    if (node == null) return;
    inOrder(node.left, k);
    cnt++;						//遍历到底之后开始算起
    if (cnt == k) {
        val = node.val;
        return;
    }
    inOrder(node.right, k);
}
```

分治解法：类似于二分查找

需要**先计算左子树的节点个数**，记为 n，然后有三种情况。

- n 加 1 等于 k，那就说明当前根节点就是我们要找的。

- n 加 1 小于 k，那就说明第 k 小的数一定在右子树中，只需要递归的在右子树中寻找第 k - n - 1 小的数即可。

- n 加 1 大于 k，那就说明第 k 小个数一定在左子树中，我们只需要递归的在左子树中寻找第 k 小的数即可。

```java
//0ms、100%
class Solution {
    public int kthSmallest(TreeNode root, int k) {//总共20个节点吧
        int cnt = nodeCount(root.left);		//cnt:9			右边有10个
        if (cnt + 1 == k) {					
            return root.val;
        } else if (cnt + 1 < k) {			//10 < k=12
            return kthSmallest(root.right, k - cnt - 1);//10 < k=12，12-9-1=2
        }
        return kthSmallest(root.left, k);
    }
    int nodeCount(TreeNode root) {
        if (root == null) return 0;
        return 1 + nodeCount(root.left) + nodeCount(root.right);
    }
}
```

### 3. BST - 累加节点值

Convert BST to Greater Tree (Easy)

[Leetcode](https://leetcode.com/problems/convert-bst-to-greater-tree/description/) / [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

```js
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13
Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

先遍历右子树。

```java
private int sum = 0;

public TreeNode convertBST(TreeNode root) {
    traver(root);
    return root;
}

void traver(TreeNode node) {
    if (node == null) return;
    traver(node.right);			//先遍历右子树。
    sum += node.val;
    node.val = sum;
    traver(node.left);
}
```

### 4. BST - 最近公共祖先

235\. Lowest Common Ancestor of a Binary Search Tree (Easy)

[Leetcode](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) / [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root.val > p.val && root.val > q.val) 
        return lowestCommonAncestor(root.left, p, q);
    if (root.val < p.val && root.val < q.val) 
        return lowestCommonAncestor(root.right, p, q);
    return root;
}
```

### 5. 二叉树 - 最近公共祖先

236\. Lowest Common Ancestor of a Binary Tree (Medium) 

[Leetcode](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/) / [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        //后序遍历
        if(left == null && right == null) return null; // 1.
        if(left == null) return right; // 3.
        if(right == null) return left; // 4.
        return root; // 2. if(left != null and right != null)
        
        //1合并到3、4
        //if(left == null) return right;
        //if(right == null) return left;
        //return root;

        //也可以再合并
        //return left == null ? right : right == null ? left : root;
    }
}
```

### 6. 有序数组构造 - BST

108\. Convert Sorted Array to Binary Search Tree (Easy)

[Leetcode](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/) / [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

给你一个整数**数组** nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return dfs(nums, 0, nums.length - 1);
    }
    TreeNode dfs(int[] nums, int l, int h) {
        if (l > h) return null;
        int mid = l + (h - l) / 2;				
        TreeNode root = new TreeNode(nums[mid]);// 以升序数组的中间元素作为根节点 root。
        root.left = dfs(nums, l, mid - 1);		// 递归的构建 root 的左子树与右子树。
        root.right = dfs(nums, mid + 1, h);
        return root;
    }
}
```

### 7. 有序链表构造 - BST

109\. Convert Sorted List to Binary Search Tree (Medium)

[Leetcode](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/) / [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

给定一个**单链表**，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return new TreeNode(head.val);
        // 快慢指针找中心节点
        ListNode s = head, f = head, preMid = null;
        while (f != null && f.next != null) {
            preMid = s;
            s = s.next;
            f = f.next.next;
        }
        preMid.next = null;						//断开链表
        TreeNode root = new TreeNode(s.val);	//升序链表的中间元素作为根节点
        root.left = sortedListToBST(head);		//构建 root 的左子树
        root.right = sortedListToBST(s.next);	//构建 root 的右子树
        return root;
    }
}
```

参考类似题目：

#### [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

```
输入：[1,2,3,4,5]		输出：3
输入：[1,2,3,4,5,6]	输出：4 由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
```

```java
ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

### 8. 两数之和 - BST

653\. Two Sum IV - Input is a BST (Easy)

[Leetcode](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/) / [653. 两数之和 IV - 输入 BST](https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst/)

```html
Input: Target = 9
    5
   / \
  3   6
 / \   \
2   4   7
Output: True
```

**中序遍历存集合+双指针**，中序遍历树，时间O(N)

应该注意到，这一题不能用分别在左右子树两部分来处理这种思想，因为两个待求的节点可能分别在左右子树中。

```java
public boolean findTarget(TreeNode root, int k) {
    List<Integer> nums = new ArrayList<>();
    inOrder(root, nums);
    int i = 0, j = nums.size() - 1;
    while (i < j) {
        int sum = nums.get(i) + nums.get(j);
        if (sum == k) return true;
        if (sum < k) i++;
        else j--;
    }
    return false;
}
void inOrder(TreeNode root, List<Integer> nums) {
    if (root == null) return;
    inOrder(root.left, nums);
    nums.add(root.val);
    inOrder(root.right, nums);
}
```

BFS + HashSet

```java
public boolean findTarget(TreeNode root, int k) {
    Set<Integer> set = new HashSet();
    LinkedList<TreeNode> queue = new LinkedList();
    queue.add(root);
    while (!queue.isEmpty()) {
        TreeNode node = queue.remove();
        if (set.contains(k - node.val)) return true;
        set.add(node.val);
        if (node.left != null) queue.add(node.left);
        if (node.right != null) queue.add(node.right); //先放左还是右都可以
    }
    return false;
}
```

先序遍历 + Set

```java
public boolean findTarget(TreeNode root, int k) {
    Set<Integer> set = new HashSet();
    return find(root, k, set);
}
boolean find(TreeNode root, int k, Set<Integer> set) {
    if (root == null) return false;
    if (set.contains(k - root.val)) return true;
    set.add(root.val);
    return find(root.left, k, set) || find(root.right, k, set);
}
```

### 9. BST - 两个节点之差的最小绝对值

530\. Minimum Absolute Difference in BST (Easy)

[Leetcode](https://leetcode.com/problems/minimum-absolute-difference-in-bst/description/) / [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

```js
Input:
   1
    \
     3
    /
   2
Output: 1
```

利用二叉查找树的中序遍历为有序的性质，**计算中序遍历中临近的两个节点之差的绝对值，取最小值。**

```java
private int minDiff = Integer.MAX_VALUE;
private TreeNode preNode = null;

public int getMinimumDifference(TreeNode root) {
    inOrder(root);
    return minDiff;
}

void inOrder(TreeNode node) {
    if (node == null) return;
    
    inOrder(node.left);
    
    if (preNode != null) minDiff = Math.min(minDiff, node.val - preNode.val);
    preNode = node;
    
    inOrder(node.right);
}
```

### 10. BST - 众数

501\. Find Mode in Binary Search Tree (Easy)

[Leetcode](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/) / [501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

答案可能不止一个，也就是有多个值出现的次数一样多。如果众数超过1个，不需考虑输出顺序。

```java
private int curCnt = 1;
private int maxCnt = 1;
private TreeNode preNode = null;//前指针，需要指向中序遍历的前一个较小的节点

public int[] findMode(TreeNode root) {
    List<Integer> maxCntNums = new ArrayList<>();
    inOrder(root, maxCntNums);
    int[] ret = new int[maxCntNums.size()];
    int idx = 0;//集合转数组
    for (int num : maxCntNums) {
        ret[idx++] = num;
    }
    return ret;
}

void inOrder(TreeNode node, List<Integer> nums) {
    if (node == null) return;
    
    inOrder(node.left, nums);
    
    if (preNode != null) {						//初始为null，注意判断null
        if (preNode.val == node.val) curCnt++;  //累加一个连续相同出现的数的次数
        else curCnt = 1;						//出现间断置位1
    }
    if (curCnt > maxCnt) {
        maxCnt = curCnt;						//更新众数的最值
        nums.clear();							//清空之前同的相同频率众数集合
        nums.add(node.val);						//加入新的更高频率的众数
    } else if (curCnt == maxCnt) {				//如果出现频率相同的众数，那就添加到众数集合
        nums.add(node.val);
    }
    preNode = node;								//更新下滚动变量
    
    inOrder(node.right, nums);
}
```

11. 前序遍历构造二叉搜索树

[1008. 前序遍历构造二叉搜索树](https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/)

题解：[前序遍历构造二叉搜索树 - 前序遍历构造二叉搜索树 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/solution/jian-kong-er-cha-shu-by-leetcode/) 

## Trie

<div align="center"> <img src="https://cs-notes-1256109796.cos.ap-guangzhou.myqcloud.com/5c638d59-d4ae-4ba4-ad44-80bdc30f38dd.jpg"/> </div><br>
Trie，又称**前缀树**或**字典树**，用于判断字符串是否存在或者是否具有某种**字符串前缀。**

### 1. 实现一个 Trie

208\. Implement Trie (Prefix Tree) (Medium)

[Leetcode](https://leetcode.com/problems/implement-trie-prefix-tree/description/) / [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```js
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[	 [],["apple"],["apple"],  ["app"],	    ["app"],  ["app"],   ["app"]]
输出
[  null,     null, 	   true, 	false, 		   true, 	 null, 	   true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

库函数

```java
//900ms，5%
class Trie {
    String data = ",";
    public Trie() {}
    
    public void insert(String word) {
        data = data+word+",";
    }
    
    public boolean search(String word) {
        return data.contains(","+word+",");
    }
    
    public boolean startsWith(String prefix) {
        return data.contains(","+prefix);
    }
}
```

```java
//49ms,35%
class Trie {

    private class Node {
        Node[] childs = new Node[26];		//存a-z
        boolean isLeaf;
    }

    private Node root = new Node();

    public Trie() {}

    public void insert(String word) { insert(word, root); }

    private void insert(String word, Node node) {
        if (node == null) return;
        if (word.length() == 0) {	//word递进到头了
            node.isLeaf = true;		//word在树的尽头节点标记下是叶子节点
            return;					//退出
        }
        int index = indexForChar(word.charAt(0));
        if (node.childs[index] == null) {		//必须判null	
            node.childs[index] = new Node();
        }
        insert(word.substring(1), node.childs[index]);			//word、node向前、下递进一个
    }

    public boolean search(String word) { return search(word, root); }
        
    private boolean search(String word, Node node) {
        if (node == null) return false;
        if (word.length() == 0) return node.isLeaf;	//word匹配到最后一个，且node到尽头
        int index = indexForChar(word.charAt(0));
        return search(word.substring(1), node.childs[index]);	//几乎统一的递归递进
    }

    public boolean startsWith(String prefix) { return startWith(prefix, root); }
        
    private boolean startWith(String prefix, Node node) {
        if (node == null) return false;
        if (prefix.length() == 0) return true;		//和search的区别：这里匹配到直接true
        int index = indexForChar(prefix.charAt(0));
        return startWith(prefix.substring(1), node.childs[index]);
    }

    private int indexForChar(char c) { return c - 'a'; }
    
}
```

```js
["Trie","insert","insert","insert","insert","insert","insert","search","search","search"]
[	 [], ["app"],["apple"],["beer"],["add"], ["jam"],["rental"],["apps"],["app"], ["ad"]]

实际错误结果
[  null,    null,     null,    null,   null,	null,	  null,	   false,  false,  false]
正确结果
[  null,	null,	  null,	   null,   null,	null,	  null,	   false, *true*,  false]
```

### 2. 实现一个 Trie，用来求前缀和

677\. Map Sum Pairs (Medium)

[Leetcode](https://leetcode.com/problems/map-sum-pairs/description/) / [677. 键值映射](https://leetcode-cn.com/problems/map-sum-pairs/)

实现一个 MapSum 类，支持两个方法，insert 和 sum：

- MapSum() 初始化 MapSum 对象
- void insert(String key, int val) 插入 key-val 键值对，字符串表示键 key ，整数表示值 val 。如果键 key 已经存在，那么原来的键值对将**被替代**成新的键值对。
- int sum(string prefix) 返回所有以该前缀 prefix 开头的键 key 的值的总和。

```js
Input: insert("apple", 3), 	 Output: Null
Input: sum("ap"), 		     return 3 (apple = 3)
Input: insert("app", 2),	 Output: Null
Input: sum("ap"), 			 return 5 (apple + app = 3 + 2 = 5)
```

```java
//15ms、68%
class MapSum {

    private class Node {
        Node[] child = new Node[26];
        int value;
    }

    private Node root = new Node();

    public MapSum() {}

    public void insert(String key, int val) { insert(key, root, val); }

    private void insert(String key, Node node, int val) {
        if (node == null) return;
        if (key.length() == 0) {
            node.value = val;			//到了尽头该下尽头的value
            return;
        }
        int index = indexForChar(key.charAt(0));
        if (node.child[index] == null) {
            node.child[index] = new Node();
        }
        insert(key.substring(1), node.child[index], val);
    }

    public int sum(String prefix) { return sum(prefix, root); }

    private int sum(String prefix, Node node) {
        if (node == null) return 0;
        if (prefix.length() != 0) {
            int index = indexForChar(prefix.charAt(0));
            return sum(prefix.substring(1), node.child[index]);
        }
        int sum = node.value;
        for (Node child : node.child) {	//复杂
            sum += sum(prefix, child);
        }
        return sum;
    }

    private int indexForChar(char c) {
        return c - 'a';
    }
}
```


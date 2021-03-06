# Leetcode 题解 - 链表
<!-- GFM-TOC -->

* [Leetcode 题解 - 链表](#leetcode-题解---链表)
    * [1. 找出两个链表的交点](#1-找出两个链表的交点)
    * [2. 链表反转](#2-链表反转)
    * [3. 归并两个有序的链表](#3-归并两个有序的链表)
    * [4. 从有序链表中删除重复节点](#4-从有序链表中删除重复节点)
    * [5. 删除链表的倒数第 n 个节点](#5-删除链表的倒数第-n-个节点)
    * [6. 交换链表中的相邻结点](#6-交换链表中的相邻结点)
    * [7. 链表求和](#7-链表求和)
    * [8. 回文链表](#8-回文链表)
    * [9. 分隔链表](#9-分隔链表)
    * [10. 链表元素按奇偶聚集](#10-链表元素按奇偶聚集)
<!-- GFM-TOC -->


链表是空节点，或者有一个值和一个指向下一个链表的指针，因此很多链表问题可以用递归来处理。

##  1. 相交链表

简单：[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

例如以下示例中 A 和 B 两个链表相交于 c1，找出两个单链表相交的起始节点。不存在交点则返回 null。

```html
A:          a1 → a2					 要求时间复杂度为 O(N)，空间复杂度为 O(1)。
                    ↘
                      c1 → c2 → c3
                    ↗
B:    b1 → b2 → b3
```

设 A 的长度为 a + c，B 的长度为 b + c，其中 c 为尾部公共部分长度，可知 a + c + b = b + c + a。

当访问 A 链表的指针访问到链表尾部时，令它从链表 B 的头部开始访问链表 B；同样地，当访问 B 链表的指针访问到链表尾部时，令它从链表 A 的头部开始访问链表 A。这样就能控制访问 A 和 B 两个链表的指针能同时访问到交点。如果不存在交点，那么 a + b = b + a，以下实现代码中 l1 和 l2 会同时为 null，从而退出循环。

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {	//双指针
    ListNode l1 = headA, l2 = headB;
    while (l1 != l2) {
        l1 = (l1 == null) ? headB : l1.next;
        l2 = (l2 == null) ? headA : l2.next;
    }
    return l1;
}
```

```c++
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {		//双指针
    ListNode *l1 = headA, *l2 = headB;
    while (l1 != l2) {
        l1 = l1 != nullptr ? l1->next : headB;
        l2 = l2 != nullptr ? l2->next : headA;
    }
    return l1;
}
```

```c
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {		//哈希表
    unordered_set<ListNode*> visited;
    ListNode *tem = headA;
    while (tem != nullptr) {						//A全存里面
        visited.insert(tem);
        tem = tem->next;							//unordered_set<> 添加：insert()
    }
    tem = headB;
    while (tem != nullptr) {						//遍历B，第一个包含的节点就是结果
        if (visited.count(tem)) return tem;			//判断set是否存在：count()
        tem = tem->next;
    }
    return nullptr;
}
```

如果只是判断是否存在交点，那么就是另一个问题，即 [编程之美 3.6]() 的问题。有两种解法：

- 把第一个链表的结尾连接到第二个链表的开头，看第二个链表是否存在环；
- 或者直接比较两个链表的最后一个节点是否相同。

##  2. 反转链表

简单： [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```java
public ListNode reverseList(ListNode head) {				//递归
    if (head == null || head.next == null) return head;
    ListNode next = head.next;
    ListNode newHead = reverseList(next);					//newHead 缓存子问题结果
    next.next = head;
    head.next = null;
    return newHead;
}
```

```c
ListNode* reverseList(ListNode* head) {						//递归
    if (!head || !head->next) return head;
    ListNode* newHead = reverseList(head->next);			
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

```java
public ListNode reverseList(ListNode head) {				//头插法
    ListNode newHead = new ListNode(-1);
    while (head != null) {
        ListNode next = head.next;
        head.next = newHead.next;
        newHead.next = head;
        head = next;
    }
    return newHead.next;
}
```

```c
ListNode* reverseList(ListNode* head) {						//头插
    ListNode* pre = nullptr;							
    ListNode* cur = head;									//不能逗号合并
    while (cur) {
        ListNode* next = cur->next;
        cur->next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}
```

##  3. 合并两个有序的链表

简单：[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {		//递归
    if (l1 == null) return l2;
    if (l2 == null) return l1;
    if (l1.val < l2.val) {										//条件定义好递归的方向，
        l1.next = mergeTwoLists(l1.next, l2);					
        return l1;												//遍历到尽头	回过头来进行连接
    } else {
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
}
```

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {		//迭代
    ListNode dum = new ListNode(0);								//dum.next为最后的头
    ListNode cur = dum;			   								//dum不动，让cur去移动。
    while(l1 != null && l2 != null) {
        if(l1.val < l2.val) {
            cur.next = l1;
            l1 = l1.next;
        }
        else {				//else if (l1.val >= l2.val)								
            cur.next = l2;
            l2 = l2.next;								
        }
        cur = cur.next;
    }
    cur.next = l1 != null ? l1 : l2;							//处理未合并完成的一支
    return dum.next;
}
```

##  4. 删除排序链表中的重复元素

简单：[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

```html
输入：head = [1,1,2,3,3]	输出：[1,2,3]
```

```c
ListNode* deleteDuplicates(ListNode* head) {					//迭代
    if (!head) return head;
    ListNode* cur = head;
    while (cur->next) {
        if (cur->val == cur->next->val) cur->next = cur->next->next;
        else cur = cur->next;
    }
    return head;
}
```

```java
public ListNode deleteDuplicates(ListNode head) {				//递归
    if (head == null || head.next == null) return head;
    head.next = deleteDuplicates(head.next);
    return head.val == head.next.val ? head.next : head;
}
```

##  5. 删除链表的倒数第 n 个节点

中等：[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```html
输入：head = [1,2,3,4,5], n = 2	输出：[1,2,3,5]
```

```java
public ListNode removeNthFromEnd(ListNode head, int n) {		//快慢双指针
    ListNode fast = head;
    while (n-- > 0) fast = fast.next;
    if (fast == null) return head.next;
    ListNode slow = head;
    while (fast.next != null) {
        fast = fast.next;
        slow = slow.next;
    }
    slow.next = slow.next.next;
    return head;
}
```

```c
ListNode* removeNthFromEnd(ListNode* head, int n) {				//快慢双指针
    ListNode* dum = new ListNode(0, head);
    ListNode* f = head;
    ListNode* s = dum;
    for (int i = 0; i < n; ++i) f = f->next;
    while (f) {
        f = f->next;
        s = s->next;
    }
    s->next = s->next->next;
    ListNode* ans = dum->next;
    delete dum;													//new后要删除
    return ans;
}
```

```c
ListNode* removeNthFromEnd(ListNode* head, int n) {				//c++栈
    ListNode* dum = new ListNode(0, head);
    stack<ListNode*> stk;
    ListNode* cur = dum;
    while (cur) {
        stk.push(cur);
        cur = cur->next;
    }
    for (int i = 0; i < n; ++i) stk.pop();
    ListNode* pre = stk.top();
    pre->next = pre->next->next;
    ListNode* ret = dum->next;									//最后
    delete dum;
    return ret;
}
```

##  6. 两两交换链表中的节点

中等： [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) 		递归参考： [递归，两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/solution/hua-jie-suan-fa-24-liang-liang-jiao-huan-lian-biao/) 

```javascript
输入：head = [1,2,3,4]  输出：[2,1,4,3]	不能修改结点的 val 值，O(1) 空间复杂度。
```

```java
public ListNode swapPairs(ListNode head) {		//迭代
    ListNode dum = new ListNode(-1);
    dum.next = head;				//dum.next指向1
    ListNode pre = dum;				//pre指向-1
    //head!=null && head.next.next!=null	1\3
    while (pre.next != null && pre.next.next != null) {
        ListNode l1 = pre.next;		//1
        ListNode l2 = pre.next.next;//2

        ListNode l3 = l2.next;		//next=3
        l1.next = l3;				//1指向了3
        l2.next = l1;				//2指向了1
        pre.next = l2;				//pre指向了2			现在是   -1	2	1	3
        pre = l1;					//pre步进到1，步进2格	 原本pre是-1		pre
    }
    return dum.next;				//返回的2
}
```

```java
public ListNode swapPairs(ListNode head) {//1  //递归		例如：1->2->3->null
    if(head == null || head.next == null) return head;
    ListNode next = head.next;			  //2
    head.next = swapPairs(next.next);	  //1->3
    next.next = head;					  //2指向1
    return next;						  //返回2
}
```

##  7. 链表求和

中等：[445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

```html
输入：(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)	输出：7 -> 8 -> 0 -> 7	 解释：7243+564=7807
```

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {	//题目要求：不能修改原始链表。
    LinkedList<Integer> l1Stack = buildStack(l1);			//思路：栈+尾插法
    LinkedList<Integer> l2Stack = buildStack(l2);			
    ListNode dum = new ListNode(-1);
    int carry = 0;
    while (!l1Stack.isEmpty() || !l2Stack.isEmpty() || carry != 0) {
        int x = l1Stack.isEmpty() ? 0 : l1Stack.pop();		//c++的 || 也可换成 or
        int y = l2Stack.isEmpty() ? 0 : l2Stack.pop();

        int sum = x + y + carry;
        carry = sum / 10;
        sum = sum % 10;		  //Modulo：取余

        ListNode cur = new ListNode(sum);
        cur.next = dum.next;  //尾插法构建链表，最新的节点在链表头部，需要dum节点。
        dum.next = cur;
    }
    return dum.next;
}

private LinkedList<Integer> buildStack(ListNode head) {
    LinkedList<Integer> stack = new LinkedList<>();
    while (head != null) {
        stack.push(head.val);
        head = head.next;
    }
    return stack;
}
```

##  8. 回文链表

简单：[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

```javascript
输入: 1->2	输出: false	输入: 1->2->2->1	输出: true	要求： O(n) 时间和 O(1) 空间
```

```java
ListNode tem;	
public boolean isPalindrome(ListNode head) {		//递归
    tem = head;
    return check(head);
}
private boolean check(ListNode head) {
    if (head == null) return true;	
    boolean res = check(head.next) && (tem.val == head.val);
    tem = tem.next;
    return res;
}
```

```java
public boolean isPalindrome(ListNode head) {	//切成两半，把后半段反转，比较两半是否相等。
    if (head == null || head.next == null) return true;
    ListNode fast = head, slow = head;			//4ms
    while (fast != null && fast.next != null) {	//快慢指针找到中点
        fast = fast.next.next;
        slow = slow.next;
    }
    if (fast != null) slow = slow.next;			//如果fast不为空，说明链表的长度是奇数个
    slow = reverse(slow);						//反转后半部分链表
    fast = head;
    while (slow != null) {
        if (fast.val != slow.val) return false;	//比较节点值是否相等
        fast = fast.next;
        slow = slow.next;
    }
    return true;
}

public ListNode reverse(ListNode head) {
    ListNode cur = head;
    ListNode pre = null;
    while (cur != null) {
        ListNode next = cur.next;
        cur.next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}
```

```java
public boolean isPalindrome(ListNode head) {		//栈实现
    if (head == null) return true;					//只需拿链表的后半部分和前半部分比较
    ListNode cur = head;	
    Stack<Integer> stack = new Stack();
    int len = 0;									//链表的长度
    while (cur != null) {							//把链表节点的值存放到栈中
        stack.push(cur.val);
        cur = cur.next;
        len++;
    }
    len >>= 1;										//len长度除以2
    while (len-- >= 0) {							//然后再出栈比较
        if (head.val != stack.pop()) return false;
        head = head.next;
    }
    return true;
}
```

##  9. 分隔链表

中等：[725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

```javascript
输入: root = [1, 2, 3], k = 5	输出: [[1],[2],[3],[],[]]
输入:root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3 输出: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]	解释: 要前长后短，每部分的长度应该尽可能的相等: 任意两部分的长度差距不能超过 1，可能部分为 null
```

```java
public ListNode[] splitListToParts(ListNode root, int k) {
    int N = 0;				
    ListNode cur = root;
    while (cur != null) {		 //求下链表长度 N
        N++;
        cur = cur.next;
    }							 //例如：10个节点，1~10，k=3	要分成4,3,3的结构
    int mod = N % k;			 //10%3=1
    int size = N / k;			 //10/3=3
    ListNode[] ret = new ListNode[k];
    cur = root;					 //数组的初始化，默认不存节点就是null
    for (int i = 0; cur != null && i < k; i++) {  //只是执行3次
        ret[i] = cur;							  //只用存一个头，后边会断开
        int curSize = size + (mod-- > 0 ? 1 : 0); //3+1=4	3+0=3，让开始分段比后边长1个
        for (int j = 0; j < curSize - 1; j++) {	  //3次		2次
            cur = cur.next;		 //cur=4					
        }
        ListNode next = cur.next;//next=5	缓存一下
        cur.next = null;		 //断开链表
        cur = next;				 //和缓存对接
    }
    return ret;
}
```

##  10. 链表元素按奇偶聚集

中等：[328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

```javascript
输入: 2->1->3->5->6->4->7->NULL	输出: 2->3->6->7->1->5->4->NULL
解释：节点为奇、偶、奇、偶、奇...的初始顺序，结果保持原始相对顺序，要求时间O(N)，空间O(1)
```

定义奇、偶链表，遍历原链表，将节点交替插入到奇链表和偶链表（尾插法），最后将偶链表拼接在奇链表后面。

```java
public ListNode oddEvenList(ListNode head) {
    if (head == null) return head;		//不加能过，最好加上。
    ListNode jHead = new ListNode();	//分别定义奇偶链表的 虚拟头结点
    ListNode oHead = new ListNode();
    ListNode curJ = jHead;				//虚拟头结点不动，使用cur来移动。
    ListNode curO = oHead;
    boolean isJ = true;
    while (head != null) {
        if (isJ) {
            curJ.next = head;		 	//尾插法
            curJ = curJ.next;
        } else {
            curO.next = head;
            curO = curO.next;
        }
        head = head.next;
        isJ = !isJ;
    }
    curJ.next = oHead.next;				//奇链表后面拼接上偶链表
    curO.next = null;					//偶链表的next设置为null
    return jHead.next;
}
```

```java
public ListNode oddEvenList(ListNode head) {
    if (head == null) return head;		//官解：但不好懂，有图
    ListNode oHead = head.next;			//缓存下偶数编号的头
    ListNode curJ = head;				//奇头	odd number	奇数
    ListNode curO = oHead;				//偶头	even		偶数
    while (curO != null && curO.next != null) {
        curJ.next = curO.next;//奇
        curJ = curJ.next;

        curO.next = curJ.next;//偶
        curO = curO.next;
    }
    curJ.next = oHead;
    return head;
}
```

## todo+++++++++++++++++++++++++++

## todo+++++++++++++++++++++++++++

## todo+++++++++++++++++++++++++++

## 11. 合并K个升序链表

困难：[23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

```java
输入：lists = [[1,4,5],[1,3,4],[2,6]]	输出：1->1->2->3->4->4->5->6	实则不难，合并两链表的升级
```

```java
public ListNode mergeKLists(ListNode[] lists) {
    Queue<ListNode> pq = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
    for (ListNode node: lists) if(node != null) pq.offer(node);
    ListNode dumHead = new ListNode(-1);	//pq 的顺序定义这里不写会报错，存引用类型的缘故
    ListNode tail = dumHead;
    while(!pq.isEmpty()) {
        ListNode minNode = pq.poll();
        tail.next = minNode;
        tail = minNode;
        if (minNode.next != null) pq.offer(minNode.next);
    }
    return dumHead.next;
}
```


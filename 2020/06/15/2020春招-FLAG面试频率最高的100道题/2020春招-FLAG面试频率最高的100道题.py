# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-03-20 12:08:11
@LastEditTime: 2020-06-15 22:08:13
@Update: 
'''

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

"""
@param matrix: matrix, a list of lists of integers
@param target: An integer
@return: a boolean, indicate whether matrix contains target
@note: https://www.lintcode.com/problem/search-a-2d-matrix/description?_from=ladder&&fromId=131
"""
def searchMatrix(matrix, target):

    def bs(array, target):
        l, r = 0, len(array) - 1
        while l <= r:
            m = l + (r - l) // 2

            if array[m] < target:
                l = m + 1
            elif array[m] > target:
                r = m - 1
            else:
                l = m + 1
        return r
    
    # 先搜行
    firstCol = [row[0] for row in matrix]
    row = bs(firstCol, target)

    if row >= len(firstCol):
        return False
    if firstCol[row] == target:
        return True
    
    # 再搜列
    rowthRow = matrix[row]
    col = bs(rowthRow, target)
    
    if col >= len(rowthRow):
        return False
    if rowthRow[col] == target:
        return True

    return False

"""
@param numbers: An array of Integer
@param target: target = numbers[index1] + numbers[index2]
@return: [index1, index2] (index1 < index2)
@note: https://www.lintcode.com/problem/two-sum/description?_from=ladder&&fromId=133
"""
def twoSum(numbers, target):

    n = len(numbers)
    # 绑定数字和标号
    numIdx = sorted(zip(numbers, range(n)), key=lambda x: x[0])

    # 左右指针
    l, r = 0, n - 1
    while l < r:
        ts = numIdx[l][0] + numIdx[r][0]
        # 找到两个数
        if ts == target:
            break
        # 两数之和偏小，左指针向右
        elif ts < target:
            l += 1
        # 两数之和偏大，右指针向左
        elif ts > target:
            r -= 1
    
    # 未找到
    if l >= r: return [None, None]

    # 返回排序后的索引
    index1, index2 = numIdx[l][1], numIdx[r][1]
    if index1 > index2:
        index1, index2 = index2, index1
    return index1, index2

"""
@param root: A Tree
@return: Level order a list of lists of integer
@example:
    root = TreeNode(1)
    root.right = TreeNode(2)
    root.right.left = TreeNode(3)
    root.right.right = TreeNode(4)
    levelOrder(root)
@note: https://www.lintcode.com/problem/binary-tree-level-order-traversal/description?_from=ladder&&fromId=102
"""
def levelOrder(root):
    
    # notification: empty tree
    if root is None:
        return []

    # 初始化队列，用`|`标记一层结束
    # 如果无需将层元素分开，无需该记号，队列即可
    from collections import deque
    q = deque([root, '|'])
    
    layers = []; layer = []
    while len(q) > 0:
        # 出队列
        front = q.popleft()

        # 到达层结束标记
        if front == '|':
            layers += [layer]
            layer = []
            # 重要：判断队列元素个数
            # 防止重复`|`出入队列造成死循环
            if len(q) > 0:
                q.extend('|')
            continue
        
        # 未到达层末标记，孩子入队列
        layer += [front.val]
        if front.left:
            q.extend([front.left])
        if front.right:
            q.extend([front.right])
    
    return layers


"""
@param l1: ListNode l1 is the head of the linked list
@param l2: ListNode l2 is the head of the linked list
@return: ListNode head of linked list
@example:
    l1 = ListNode(3)
    l1.next = ListNode(5)
    l1.next.next = ListNode(7)
    l1.next.next.next = ListNode(8)
    l1.next.next.next.next = ListNode(10)
    l2 = ListNode(3)
    l2.next = ListNode(4)
    l2.next.next = ListNode(6)
    l2.next.next.next = ListNode(9)
    l2.next.next.next.next = ListNode(11)
    head = mergeTwoLists(l1, l2)
@note: https://www.lintcode.com/problem/merge-two-sorted-lists/description?_from=ladder&&fromId=126
"""
def mergeTwoLists(l1, l2):
    head = ListNode(None)
    node = head
    # 归并排序中，合并两个数组的思路
    while l1 and l2:
        if l1.val < l2.val:
            node.next = l1
            l1 = l1.next
        else:
            node.next = l2
            l2 = l2.next
        node.next.next = None
        node = node.next
    
    node.next = l1 if l1 else l2
    return head.next

"""
@param s: a string
@return: return a integer
@note: https://www.lintcode.com/problem/longest-valid-parentheses/description?_from=ladder&&fromId=131
    注意`"(-(())-(-((()()))"`的情况，不能只一味累加有效括号长度，需要在“合适时机”清除累积变量
    第一种解法只是累计有效括号对，没有对有效括号串进行分隔；
    第二种用数组标记每位，有效的置为True，然后统计连续True的最长长度。
"""
# def longestValidParentheses(s):
#     longestLen = 0

#     # 双端队列用作栈
#     from collections import deque
#     q = deque()

#     # 用于累计有效括号数，如`(()())`的情况下，需要累计`()()`的长度4
#     length = 0
#     for i, c in enumerate(s):
#         # 左括号，入栈等待处理
#         if c == '(':
#             q.extend([c])

#         # 右括号，始终不入栈
#         elif c == ')':
#             # 出现`)`时栈空，则无效
#             if len(q) == 0:
#                 length = 0
#                 continue
#             # 出现`)`时栈顶为`(`，则有效
#             if q[-1] == '(':
#                 q.pop()
#                 length += 2
        
#         # 更新最长长度
#         longestLen = max(longestLen, length)
#     return longestLen

def longestValidParentheses(s):

    # 双端队列用作栈
    from collections import deque
    q = deque()

    # 数组统计有效括号
    a = [False for i in range(len(s))]

    for i, c in enumerate(s):
        # 左括号，入栈等待处理(包括索引和括号)
        if c == '(':
            q.extend([(i, c)])

        # 右括号，始终不入栈
        elif c == ')':
            # 出现`)`时栈空，则无效
            if len(q) == 0:
                continue
            
            # 出现`)`时栈顶为`(`，则有效
            # 置数组元素为真(一对)
            if q[-1][-1] == '(':
                a[i] = True
                a[q[-1][0]] = True
                q.pop()
    
    # 统计最长连续为True的长度
    longestLen = 0
    length = 0
    for i, b in enumerate(a):
        if b == True:
            length += 1
        elif b == False:
            length = 0
        longestLen = max(longestLen, length)
    return longestLen

"""
@param s: A string
@return: whether the string is a valid parentheses
@note: https://www.lintcode.com/problem/valid-parentheses/description?_from=ladder&&fromId=137
"""
def isValidParentheses(s):
    
    # 双端队列模拟栈
    from collections import deque
    q = deque()
    # 有效括号对，用于反查左括号
    parentheses = {')': '(', ']': '[', '}': '{'}

    for i, c in enumerate(s):
        # 左半括号，入栈
        if c in parentheses.values():
            q.extend([c])
        
        # 右半括号，始终不入栈
        else:
            # 栈为空，无效
            if len(q) == 0:
                return False
            
            # 匹配成功，左括号出栈
            if parentheses[c] == q[-1]:
                q.pop()
                continue
            # 匹配失败
            else:
                return False
    
    return True if len(q) == 0 else False

"""
@param grid: a boolean 2D matrix
@return: an integer
@note: https://www.lintcode.com/problem/number-of-islands/description?_from=ladder&&fromId=131
"""
def numIslands(grid):

    # 泛洪填充
    def flood(grid, m, n, i, j, srcNum, dstNum):
        # 终止条件1：越界
        if i < 0 or i >= m or j < 0 or j >= n:
            return
        # 终止条件2：已到达过
        if grid[i][j] != srcNum:
            return
        
        grid[i][j] = dstNum
        # 回溯
        flood(grid, m, n, i - 1, j, srcNum, dstNum)
        flood(grid, m, n, i + 1, j, srcNum, dstNum)
        flood(grid, m, n, i, j - 1, srcNum, dstNum)
        flood(grid, m, n, i, j + 1, srcNum, dstNum)
    
    # 验证矩阵有效性
    m = len(grid)
    if m == 0: return 0
    n = len(grid[0])
    if n == 0: return 0

    # 填充标记
    flagNum = 2
    for i in range(m):
        for j in range(n):
            # 未到达过则进行填充
            if grid[i][j] == 1:
                flood(grid, m, n, i, j, 1, flagNum)
                flagNum += 1
    
    # 统计个数
    flagNums = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0: continue
            if grid[i][j] not in flagNums:
                flagNums += [grid[i][j]]
    return len(flagNums)

"""
@param n: An integer
@return: true if this is a happy number or false
@note: https://www.lintcode.com/problem/happy-number/description?_from=ladder&&fromId=131
"""
def isHappy(n):
    nums = [n]
    while True:
        # 各位求平方和
        n = sum(map(lambda x: int(x)**2, str(n)))
        # 最终为1，是快乐数
        if n == 1:
            return True
        # 数字已出现，死循环
        if n in nums:
            return False
        # 保存已出现的数字
        nums += [n]

"""
@param root: the root of binary tree
@return: the root of the maximum average of subtree
@note: https://www.lintcode.com/problem/subtree-with-maximum-average/description?_from=ladder&&fromId=78
    后序遍历，过程中计算每棵子树中节点的个数，以及对应的均值
@example:
    root = TreeNode(1)
    root.left = TreeNode(-5)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(2)
    root.right = TreeNode(11)
    root.right.left = TreeNode(4)
    root.right.right = TreeNode(-2)
    findSubtree2(root)
"""
def findSubtree2(root):
    if root is None: return root
    maxavg, maxroot = findSubtree2Core(root, float('-inf'), None)
    return maxroot

def findSubtree2Core(root, maxavg, maxroot):
    if root is None:
        return maxavg, maxroot
    
    # 累计左右子树计数、和
    count = 0; sumval = 0

    # 左子树
    if root.left:
        maxavg, maxroot = findSubtree2Core(root.left, maxavg, maxroot)
        count += root.left.count
        sumval += root.left.count * root.left.avg
    # 右子树
    if root.right:
        maxavg, maxroot = findSubtree2Core(root.right, maxavg, maxroot)
        count += root.right.count
        sumval += root.right.count * root.right.avg
    
    # 本节点
    root.count = count + 1
    root.avg = (sumval + root.val) / root.count

    # 更新最大
    return (root.avg, root) if root.avg > maxavg else (maxavg, maxroot)



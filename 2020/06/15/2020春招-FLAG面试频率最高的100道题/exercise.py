# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-03-20 12:08:11
@LastEditTime: 2020-06-19 13:42:51
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
    
def buildtree(data):
    if data is None: return None

    from collections import deque
    root = TreeNode(data[0])
    nq = deque([root])
    vq = deque(data[1:])
    
    while len(vq) > 0:
        node = nq.popleft()
        if node is None: continue
        
        # 左子节点
        val = vq.popleft()
        if val != '#':
            node.left  = TreeNode(val)
            nq.extend([node.left] )
        # 这里是为了使nq与vq对齐，否则一直popleft，会导致nq提前为空
        else:
            nq.extend([None])
        # 右子节点
        val = vq.popleft()
        if val != '#':
            node.right = TreeNode(val)
            nq.extend([node.right])
        # 这里是为了使nq与vq对齐，否则一直popleft，会导致nq提前为空
        else:
            nq.extend([None])
    
    return root

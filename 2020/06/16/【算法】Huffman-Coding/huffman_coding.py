# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2020-06-16 10:38:45
@LastEditTime: 2020-06-16 14:12:31
@Update: 
'''
from heapq import nlargest
from collections import Counter, deque

poem = \
"""
This is my prayer to thee, my lord
strike, strike at the root of penury in my heart.
Give me the strength lightly to bear my joys and sorrows.
Give me the strength to make my love fruitful in service.
Give me the strength never to disown the poor or bend my knees before insolent might.
Give me the strength to raise my mind high above daily trifles.
And give me the strength to surrender my strength to thy will with love.
"""

class HuffmanTreeNode:

    def __init__(self, weight, char=None):
        # 左右子树
        self.left = None
        self.right = None

        # 属性
        self.weight = weight    # 权重
        self.char = char        # 代表字符
        self.code = None        # 节点编码，用字符串表示
    
    def isLeaf(self):
        return self.left is None and self.right is None

class HuffmanTree:

    def __init__(self):
        self.root = None
        self.table = None
    
    def _build(self, text):
        """ 构建霍夫曼树
        Parameters:
            text: {str}
        Returns:
            node: {HuffmanTreeNode}
        """
        # 统计字符数
        counter = Counter(text)
        # 初始化节点，这里进行了一次排序使树结构尽量稳定
        kvps = sorted(counter.items(), key=lambda x: x[0])
        nodes = [HuffmanTreeNode(n / sum(counter.values()), c) \
                for c, n in kvps]
        
        # 循环构建树
        while len(nodes) > 1:
            # 选择权重最小的两个节点
            left, right = nlargest(2, nodes, key=lambda x: - x.weight)
            # 新建节点
            weight = left.weight + right.weight
            node = HuffmanTreeNode(weight)
            node.left, node.right = left, right
            
            # 移除节点
            nodes.remove(left)
            nodes.remove(right)
            # 加入节点
            nodes += [node]

        return nodes[0]
    
    def _table(self, root):
        """ 获取编码列表
        Parameters:
            None
        Returns:
            table
        Notes:
            层次遍历(BFS)得到各字符的表示
        """
        # 初始化队列
        q = deque()
        self.root.code = ''
        q.extend([self.root])

        # bfs
        table = dict()
        while len(q) > 0:
            # 队首元素
            node = q.popleft()

            # 叶子节点
            if node.isLeaf():
                table[node.char] = node.code
                continue
            # 非叶子节点
            if node.left:
                node.left.code = node.code + '0'
                q.extend([node.left])
            if node.right:
                node.right.code = node.code + '1'
                q.extend([node.right])
        return table
    
    def build(self, text):
        """
        Parameters:
            text: {str}
        Returns:
            None
        """
        self.root = self._build(text)
        self.table = self._table(self.root)

    def encode(self, text):
        """
        Parameters:
            text: {str}
        Returns:
            encoded: {int}
        """
        encoded = 0
        for i, char in enumerate(text):
            code = self.table[char]
            for c in code:
                encoded = (encoded << 1) + int(c)
        return encoded
    
    def decode(self, encoded):
        """
        Parameters:
            encoded: {int}
        Returns:
            text: {str}
        """
        text = ''
        encoded = bin(encoded)
        node = self.root
        for i, c in enumerate(encoded[2:]):
            if c == '0':
                node = node.left
            elif c == '1':
                node = node.right
            
            if node.isLeaf():
                text += node.char
                node = self.root
        return text

# 构建
tree = HuffmanTree()
tree.build(poem)
print(poem)

# 编码
encoded = tree.encode(poem)
strEncoded = bin(encoded)
print(strEncoded)
print((len(strEncoded) - 2) / 8, " Bytes")

# 解码
decoded = tree.decode(encoded)
print(decoded)
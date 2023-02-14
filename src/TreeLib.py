import math
import random
import bifurcationMathFunctions as bmf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


#Synthetic angogram class: Node
class Node:
    def __init__(self, coord, width, length, angle, bifurcProb, bifurcBigLeft):
        self.coord = coord
        self.width = width
        self.length = length
        self.angle = angle
        self.bifurcProb = bifurcProb
        self.bifurcBigLeft = bifurcBigLeft
        self.children = []

    #Creates a child node
    def createChild(self, angle, ratio):
        newWidth = self.width*ratio
        newLength = self.length*ratio
        curX = self.coord[0]
        curY = self.coord[1]
        newX = curX + math.cos(angle)*newLength
        newY = curY + math.sin(angle)*newLength
        newCoord = newX, newY
        return Node(newCoord, newWidth, newLength, angle, self.bifurcProb, self.bifurcBigLeft)

    #Creates and add either 1 or 2 children to the parent node.
    def addChildren(self):
        bifurcation = (random.random() < self.bifurcProb)
        if bifurcation:
            # alpha is the ratio between diameters small/big
            alpha = bmf.getRandomAlpha()
            # Make 2 children mirrored across an angle
            # getAllParameters returns (bigParams, smallParams).
            # The bifurcBigLeft randomises direction of biggest new artery
            if self.bifurcBigLeft < random.random():
                (ratioL, ratioR), (angleL, angleR) = bmf.getAllParameters(alpha)
            else:
                (ratioR, ratioL), (angleR, angleL) = bmf.getAllParameters(alpha)
            leftChild = self.createChild(self.angle-angleL, ratioL)
            self.children.append(leftChild)
            rightChild = self.createChild(self.angle+angleR, ratioR)
            self.children.append(rightChild)
        else:
            #Make 1 child following an angle
            singleChild = self.createChild(self.angle, 1)
            self.children.append(singleChild)


#Synthetic Angiogram class: Tree
class Tree:
    def __init__(self, x, y, width, length, angle, stopWidth, bifurcProb=0.3, bifurcBigLeft = 0.5):
        self.stopWidth = stopWidth
        self.addRoot(x,y, width, length, angle, bifurcProb, bifurcBigLeft)
        self.makeTree()

    def addRoot(self, x, y, width, length, angle, bifurcProb, bifurcBigLeft):
        self.root = Node((x,y), width, length, angle, bifurcProb, bifurcBigLeft)
    
    def growTree(self, node):
        if node.width < self.stopWidth:
            return
        else:
            node.addChildren()
            for child in node.children:
                self.growTree(child)
    
    def makeTree(self):
        self.growTree(self.root)
        

def nodeInside(cx, cy, X, Y):
    return (cx < X and cx >= 0 and cy < Y and cy >= 0)

def drawNode(node, draw):
    if len(node.children) >= 1:
        px, py = int(node.coord[0]), int(node.coord[1])
        for child in node.children:
            cx, cy = int(child.coord[0]), int(child.coord[1])
            draw.line((px, py, cx, cy), fill = 1, width=int(child.width))
            drawNode(child, draw)
    else:
        return


#Generates the tree as a 2D numpy array
def genTree(tree, dim):
    img = Image.new("1", dim)
    draw = ImageDraw.Draw(img)
    drawNode(tree.root, draw)
    img = np.array(img)
    return img

#generates and plots a tree
def drawTree(tree, dim):
    img = genTree(tree, dim) * 255
    plt.imshow(img,cmap="gray", vmin=0, vmax=255)
    plt.show()

#plot tree
#def drawTree(tree)
import math
import random
import synDataFunctionality.bifurcationMathFunctions as bmf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

def genNewLength(length, lengthRatioMean, lengthRatiostd):
    ratio = np.random.normal(lengthRatioMean, lengthRatiostd)
    while ratio < 0.3: #blood vessels do not really become smaller than this
        ratio = np.random.normal(lengthRatioMean, lengthRatiostd)
    newLen = ratio*length
    if newLen < 10: #The lengths do not become smaller than 10
        newLen = 10
    #print("ratio {0}, length {1}".format(ratio, length))
    #print(ratio*length)
    return newLen
#Synthetic angogram class: Node
class Node:
    def __init__(self, coord, width, pLength, angle, bifurcProb, bifurcBigLeft, stdMul, lengthRatioMean, lengthRatiostd):
        self.coord = coord
        self.width = width
        self.pLength = pLength
        self.angle = angle
        self.stdMul = stdMul
        self.bifurcProb = bifurcProb
        self.bifurcBigLeft = bifurcBigLeft
        self.lengthRatioMean = lengthRatioMean
        self.lengthRatiostd = lengthRatiostd
        self.children = []

    #Creates a child node
    def createChild(self, angle, ratioWidth):#, roadtravelled):
        newWidth = self.width*ratioWidth
        newLength = genNewLength(self.pLength, self.lengthRatioMean, self.lengthRatiostd)  #(self.pLength + roadtravelled)*ratio 
        curX = self.coord[0]
        curY = self.coord[1]
        newX = curX + math.cos(angle)*newLength
        newY = curY + math.sin(angle)*newLength
        newCoord = newX, newY
        return Node(newCoord, newWidth, newLength, angle, self.bifurcProb, self.bifurcBigLeft, self.stdMul, self.lengthRatioMean, self.lengthRatiostd)

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
            leftChild = self.createChild(self.angle-angleL, ratioL)#, 0)
            self.children.append(leftChild)
            rightChild = self.createChild(self.angle+angleR, ratioR)#, 0)
            self.children.append(rightChild)
        else:
            #Make 1 child following an angle
            singleChild = self.createChild(
                random.gauss(self.angle, self.angle*self.stdMul),
                0.98
                )#, self.pLength)
            self.children.append(singleChild)


#Synthetic Angiogram class: Tree
class Tree:
    def __init__(self, x, y, width, pLength, angle, stopWidth, angleStdMul = 0.0, bifurcProb=0.3, bifurcBigLeft = 0.5, lengthRatioMean = 0.95, lengthRatiostd=0.43):
        self.stopWidth = stopWidth
        self.angleStdMul = angleStdMul
        self.addRoot(x,y, width, pLength, angle, bifurcProb, bifurcBigLeft, lengthRatioMean, lengthRatiostd)
        self.makeTree()

    def addRoot(self, x, y, width, pLength, angle, bifurcProb, bifurcBigLeft, lengthRatioMean, lengthRatiostd):
        self.root = Node((x,y), width, pLength, angle, bifurcProb, bifurcBigLeft, self.angleStdMul, lengthRatioMean, lengthRatiostd)
    
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

#generates and plots a 2D float arr (with values in [0, 1])
def drawTree(arr, lab=False):
    #img = genTree(tree, dim) * 255
    if lab:
        plt.imshow(arr*255, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.show()

#plot tree
#def drawTree(tree)
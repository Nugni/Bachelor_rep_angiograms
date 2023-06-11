import math
import random
from sklearn.mixture import GaussianMixture
import syn_data_functionality.bifurcation_math_functions as bmf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

#Lognormal cdf of probability of bifurcating after 0, 1, .. linepieces. 
bifurc_cdf = [0, 0.17819338490705852, 0.4948775648691266, 0.796706906909377, 0.9509823643337217, 0.9932718619501374, 0.9994886850041991, 0.9999788076670704, 0.9999995301294428, 1]


def gen_new_length(width, length_ratio_mean, length_ratio_std): #, smallest_next_length_ratio):
    # Get ratio between current and next length
    next_length_ratio = np.random.lognormal(-2.1367, 0.6427)#np.random.normal(0.10354471, 0.15610471)#np.random.lognormal(-2.1367, 0.6427)##np.random.lognormal(-2.1367, 0.6427)#np.random.normal(0.10354471, 0.15610471) #length_ratio_mean, length_ratio_std)
    while next_length_ratio < 0.25:#3:#0.2:#0.3: #0.25: # smallest_next_length_ratio:
        next_length_ratio = np.random.lognormal(-2.1367, 0.6427)#np.random.normal(0.10354471, 0.15610471)#np.random.lognormal(-2.1367, 0.6427)#np.random.lognormal(-2.1367, 0.6427) #length_ratio_mean, length_ratio_std)
    new_length = width / next_length_ratio

    # constricting length: 10 < length < 80
    if new_length < 10:
        new_length = 10
    if new_length > 80:
        new_length = 80

    return new_length


#Synthetic angogram class: Node
class Node:
    def __init__(self, coord, width,
                 prev_length, angle,
                 bifurc_prob, bifurc_big_left,
                 stdMul,
                 length_ratio_mean, length_ratio_std,
                 counter,
                 single_child_width_ratio = 0.98):
        self.coord = coord
        self.width = width
        self.prev_length = prev_length
        self.angle = angle
        self.bifurc_prob = bifurc_prob
        self.bifurc_big_left = bifurc_big_left
        self.stdMul = stdMul
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.single_child_width_ratio = single_child_width_ratio
        self.counter = counter
        self.children = []

    # Create a child node
    def create_child(self, angle, ratio_width, counter):
        # Find new width
        new_width = self.width*ratio_width
        # Find new coordinates
        new_length = gen_new_length(new_width , self.length_ratio_mean, self.length_ratio_std)
        cur_x, cur_y = self.coord
        new_x, new_y = cur_x + math.cos(angle)*new_length, cur_y + math.sin(angle)*new_length

        return Node((new_x,new_y), new_width, new_length, angle, self.bifurc_prob, self.bifurc_big_left, self.stdMul, self.length_ratio_mean, self.length_ratio_std, counter)

    # Creates and add either 1 or 2 children to the parent node
    def add_children(self, stop_width):
        # Bifurcation should happen with a certain possibility
        bifurcation = (random.random() < bifurc_cdf[self.counter]) # bifurc_prob)
        if bifurcation:
            # Get ratio between artery diameters: 0 < small/big <= 1
            alpha = bmf.get_random_alpha()

            # bifurc_big_left randomizes direction of biggest artery
            if self.bifurc_big_left < random.random():
                (ratio_l, ratio_r), (angle_l, angle_r) = bmf.get_all_parameters(alpha)
            else:
                (ratio_r, ratio_l), (angle_r, angle_l) = bmf.get_all_parameters(alpha)

            # Make children bifurcating at an angle
            leftChild = self.create_child(self.angle-angle_l, ratio_l, 1)
            rightChild = self.create_child(self.angle+angle_r, ratio_r, 1)

            # Ensures too narrow children are not drawn
            if leftChild.width >= stop_width:
                self.children.append(leftChild)
            if rightChild.width >= stop_width:
                self.children.append(rightChild)

        else:
            # Make 1 child following an angle
            single_child = self.create_child(
                self.angle + np.random.normal(0,25)/360*2*math.pi,
                self.single_child_width_ratio, self.counter+1)
            # Ensures too narrow children are not drawn
            if single_child.width >= stop_width:
                self.children.append(single_child)


#Synthetic Angiogram class: Tree
class Tree:
    def __init__(self, x, y, width, prev_length, angle, stop_width, angle_std_mul = 0.0, bifurc_prob=0.3, bifurc_big_left = 0.5, length_ratio_mean = 0.95, length_ratio_std=0.43):
        self.stop_width = stop_width
        self.angle_std_mul = angle_std_mul
        self.add_root(x,y, width, prev_length, angle, bifurc_prob, bifurc_big_left, length_ratio_mean, length_ratio_std)
        self.make_tree()

    def add_root(self, x, y, width, prev_length, angle, bifurc_prob, bifurc_big_left, length_ratio_mean, length_ratio_std):
        self.root = Node((x,y), width, prev_length, angle, bifurc_prob, bifurc_big_left, self.angle_std_mul, length_ratio_mean, length_ratio_std, 1)

    def grow_tree(self, node):
        if node.width < self.stop_width:
            return
        else:
            node.add_children(self.stop_width)
            for child in node.children:
                self.grow_tree(child)

    def make_tree(self):
        self.grow_tree(self.root)


def node_inside(cx, cy, X, Y):
    return (cx < X and cx >= 0 and cy < Y and cy >= 0)

def draw_node(node, draw):
    if len(node.children) >= 1:
        parent_x, parent_y = int(node.coord[0]), int(node.coord[1])
        for child in node.children:
            child_x, child_y = int(child.coord[0]), int(child.coord[1])
            draw.line((parent_x, parent_y, child_x, child_y), fill = 1, width=round(child.width), joint='curve')
            #ensures bendy lines, and not 'crackled' lines
            offset = (round(child.width)-1)/2 - 1
            draw.ellipse ((child_x-offset, child_y-offset, child_x+offset, child_y+offset), fill=1)
            draw_node(child, draw)
    else:
        return


#Generates the tree as a 2D numpy array
def gen_tree(tree, dim):
    img = Image.new("1", dim)
    draw = ImageDraw.Draw(img)
    draw_node(tree.root, draw)
    img = np.array(img)
    return img

# generates and plots a 2D float arr (with values in [0, 1])
def draw_tree(arr, lab=False):
    #img = genTree(tree, dim) * 255
    if lab:
        plt.imshow(arr*255, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.show()

#plot tree
#def drawTree(tree)
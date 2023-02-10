import turtle
import random

# Initial Conditions
turtle.tracer(False)
turtle.hideturtle()
turtle.screensize(736,736)
turtle.radians()

# random values
alpha = random.random()
angleMed = 0.3
def angleVar(): return random.gauss(0,0.3)
def angleNew(): return angleMed + angleVar()
lenMed = 40
def lenVar(): return random.gauss(0,40)
def lenNew(): return lenMed + lenVar()



# Set turtles position, heading and width.
# Used to reset turtle position to 
def turtleSet(pos,angle,width):
    turtle.penup()
    turtle.setpos(pos)
    turtle.setheading(angle)
    turtle.width(width)
    turtle.pendown()

# Makes an image of angiogram-like structure
def turtleWalk(pos,angle,width):
    # Recursion Base Case
    if width == 0:
        return
    
    # Establish recursion point
    orgPos = turtle.position()
    orgAngle = turtle.heading()
    turtle.width(width)
    
    # Biggest child
    turtle.left(angleNew())
    turtle.forward(lenNew())
    turtleWalk(turtle.position(), turtle.heading(), width-1)
    
    # Reset position
    turtleSet(orgPos,orgAngle,width)
    
    # Smallest Child
    turtle.right(angleNew())
    turtle.forward(lenNew())
    turtleWalk(turtle.position(), turtle.heading(), width-1)

turtleWalk((400,400), 0, 5)


angio = turtle.getcanvas()

angio.postscript(file="test_angio.ps")

turtle.done()

turtle.bye()
#%%
import numpy as np
img = np.loadtxt("test_angio.ps")





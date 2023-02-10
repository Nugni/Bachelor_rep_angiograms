import turtle

turtle.tracer()
turtle.hideturtle()
turtle.screensize(736,736)
turtle.radians()


def turtleSet(pos,angle,width):
    turtle.pu()
    turtle.setpos(pos)
    turtle.setheading(angle)
    turtle.width(width)
    turtle.pd()
    


def turtleWalk(pos,angle,w):
    if w == 0:
        return
    
    orgPos = turtle.position()
    orgAngle = turtle.heading()
    
    turtleSet(orgPos, orgAngle, w)
    
    turtle.left(0.5)
    turtle.forward(40)
    turtleWalk(turtle.position(), turtle.heading(), w-1)
    
    turtleSet(orgPos,orgAngle,w)
    
    turtle.right(0.5)
    turtle.forward(40)
    turtleWalk(turtle.position(), turtle.heading(), w-1)

turtleWalk((400,400), 0, 5)




turtle.done()
turtle.bye()
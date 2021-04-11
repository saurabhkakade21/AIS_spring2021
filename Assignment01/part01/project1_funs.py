#AIS Programming assignment: 01 (Part 01)
#Programmer: Saurabh Jawahar Kakade 
#NAU Email ID: sk2354@nau.edu

import string

#load board function
def loadBoard(newboard):
    board = open(newboard,"r")
    new_board = []
    for i in board:
        new_board.append(i.split())
    return new_board

#print board function
def printBoard(newboard):
    for i in newboard:
        print(' '.join(map(str,i)))

#possible move function
def possibleMoves(moves,newboard):
    newpmoves = list() #possible moves list
    x = moves[0] #row
    y = moves[1] #column

    l = len(newboard)

    #top coordinates
    if (x-(x-1)==1) & (y-(y-1)==1):                 
        if (l>(x-1)>-1) & (l>(y-1)>-1):
            newpmoves = newpmoves + [(x-1,y-1)]

    if (x-(x-1)==1):                           
        if (l>(x-1)>-1) & (l>(y)>-1):
            newpmoves = newpmoves + [(x-1,y)]

    if (x-(x-1)==1) & (((y+1)-y)==1):               
        if (l>(x-1)>-1) & (l>(y+1)>-1):
            newpmoves = newpmoves + [(x-1,y+1)]
    
    #left coordinates
    if (y-(y-1)==1):                 
        if (l>(x)>-1) & (l>(y-1)>-1):
            newpmoves = newpmoves + [(x,y-1)]

    #right coordinates
    if (((y+1)-y)==1):               
        if (l>(x)>-1) & (l>(y+1)>-1):
            newpmoves = newpmoves + [(x,y+1)]
    
    #bottom coordinates
    if ((x+1)-x==1) & (y-(y-1)==1):                 
        if (l>(x+1)>-1) & (l>(y-1)>-1):
            newpmoves = newpmoves + [(x+1,y-1)]

    if ((x+1)-x==1):                           
        if (l>(x+1)>-1) & (l>(y)>-1):
            newpmoves = newpmoves + [(x+1,y)]

    if ((x+1)-x==1) & (((y+1)-y)==1):               
        if (l>(x+1)>-1) & (l>(y+1)>-1):
            newpmoves = newpmoves + [(x+1,y+1)]
    
    else:
        print("error")

    return newpmoves

#Legal moves function    
def legalMoves(m0,m1):  #m0 is all possible moves and m1 is visited moves
    l1 = set(m1)
    l2 = set(m0)
    lm = set(l2 - l1) #legal moves set
    # print(lm)
    return lm

#examine state function
def examineState(myboard,m1,m2,myDict):
    #m1=current position
    #m2=current path
    es = list() #examine state list variable
    

    for i in m2:
        x, y = i    #x=row and y=column
        es.append(myboard[x][y])
    
    x, y = m1
    
    es.append(myboard[x][y])
    
    #fes variable = final examine state 
    fes = "".join(map(str,es))

    
    if fes.lower() in myDict:
        print((fes.lower(),"Yes"))
    else:
        print((fes.lower(),"No"))
            
    
    #end of code

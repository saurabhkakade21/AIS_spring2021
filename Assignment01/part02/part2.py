
#AIS Programming assignment: 01 (Part 02)
#Programmer: Saurabh Jawahar Kakade 
#NAU Email ID: sk2354@nau.edu

import string
import time

start_prog_time = time.time()

#(top, right, bottom, left and 4 diagonal moves)
row = [-1, -1, -1, 0, 1, 0, 1, 1]   #8 moves in row
col = [-1, 1, 0, -1, -1, 1, 0, 1]   #8 moves in column

# Load Board function is used to load the boggle game board
def loadBoard(newboard):
    board = open(newboard,"r")
    print(board)
    new_board = []
    for i in board:
        new_board.append(i.split())
    return new_board

#Print Board function is used for printing the boggle game board
def printBoard(newboard):
    for i in newboard:
        print(' '.join(map(str,i)))

#Function to check the boundaries of board's row and column
def isSafe(path, traversed):
    x,y = path
    # print((0 <= x < r) and (0 <= y < c) and not traversed[x][y])
    return (0 <= x < r) and (0 <= y < c) and not traversed[x][y]


#Recursive function 
def search_Boggle_For_Words(myBoard, words, visited, i, j, move=''):

    global word_counter
    
    word_counter += 1
    
    # current position = visited
    visited[i][j] = True

    # increment move (current path) with myboard and add to words list    
    move = move + myBoard[i][j]
    words.add(move) 


    # Compare with 8 directions possible moves

    for k in range(8):
        x1 = i + row[k]
        y1 = j + col[k]
        s_path = (x1,y1)
        if isSafe(s_path,visited):
            
            search_Boggle_For_Words(myBoard, words, visited, x1, y1, move)

    # current position = not visited
    visited[i][j] = False

# Search function for words in a board
def search_The_Boggle_Board(myBoard, input):

    visited = [[False for x in range(len(myBoard))] for y in range(len(myBoard))]
    words = set()
    
    for i in range(len(myBoard)):
        for j in range(len(myBoard)):
            # consider each character as a starting point and run DFS
            search_Boggle_For_Words(myBoard, words, visited, i, j)

    list_inDict = []
    
    for word in input:
        if word in words:
            list_inDict.append(word)
            
    return list_inDict

# This function is used to create a dictionary of words based on their length
def group_By_Number_Of_Char(list_InDict):
    
    myDict = {} 
  
    for word in list_InDict: 
        if len(word) not in myDict: 
            myDict[len(word)] = [word] 
        elif len(word) in myDict: 
            myDict[len(word)] += [word] 
      
    res = []
     
    for key in sorted(myDict): 
        res.append(myDict[key]) 
      
    return res

# This is the runBoard function
def runBoard(myBoard):
    
    printBoard(myBoard)
    print("\n")
    
    
    myDict = (word.strip() for word in open("twl06.txt"))
    
    word=list(myDict)

    global word_counter
    
    word_counter = 0
    
    list_inDict=search_The_Boggle_Board(myBoard, word)
    
    # print(list_inDict)
    
    g = group_By_Number_Of_Char(list_inDict)
    
    print("Searched total of %d moves in %s seconds" %(word_counter, (time.time() - start_prog_time)))
    print("\n")
    
    print("Words found:")
    
    for i in range(len(g)):
        print("%d-letter words: " %len(g[i][1]))
        for j in range(len(g[i])):
            print(g[i][j].upper(), end=" ")
        print("\n")   
    
    print("Found %d words in total" %len(list_inDict))
    print("Alpha-sorted list words:")
    print("\n")
    
    sortedlist_inDict = sorted(list_inDict)
    
    for i in range(len(sortedlist_inDict)):
        print(sortedlist_inDict[i].upper(), end=" ")


myBoard=loadBoard('board.txt')
r=c=len(myBoard)
runBoard(myBoard)
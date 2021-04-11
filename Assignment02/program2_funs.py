import graphmaker as GraphMaker
from DRDViz import DRDViz
from node import Node

x = DRDViz()  

start = []  #start node variable
goal = []   #goal node variable
path = []   #path variable

def bfs(graph,start,end):
    queue = []
    queue.append([start])
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end:
            return path
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

    # print(graph)

  
# defination to load the graph from the given file path

def loadData(fileName):

    myData = open(fileName, "r").read().split("\n")

    mySecondData = list()

    myThirdData = list()

    for i in range(len(myData)):
        mySecondData.append(myData[i].split(","))

    for j in range(len(mySecondData)):
        myThirdData.append(myData[j].rsplit("'"))
        # print(myThirdData[j][1]+" "+myThirdData[j][3])

    myCustomDict = {}
    secondList = list()
    currSel = ''

    for i in range(len(myThirdData)):
        
        currSel = myThirdData[i][1]

        for j in range(len(myThirdData)):

            if(myThirdData[j][1] == currSel):

                secondList.append(myThirdData[j][3])

        myCustomDict[currSel] = secondList

        secondList = []

        # print(myCustomDict)

    thirdList = []

    for i in range(len(myThirdData)):

        currSel = myThirdData[i][1]
        # l = list()
        # # print(myThirdData[i][4].split(", ["))
        # print(''.join(map(str,myThirdData[i][4])))

        # for i in range(len(myThirdData)):
        #     l.append(myThirdData[i][4].split(","))

        # for j in range(len(mySecondData)):
        #     myThirdData.append(myData[j].rsplit("'"))
        

        for j in range(len(myThirdData)):

            if(myThirdData[j][3] == currSel):

                thirdList.append(myThirdData[j][1])

        # myCustomDict[currSel].append(thirdList)

        for x in range(len(thirdList)):

            myCustomDict[currSel].append(thirdList[x])

        thirdList = []

        # print(myCustomDict)


        for key, value in myCustomDict.items():

            # print(list(set(myCustomDict[key])))

            myCustomDict[key] = sorted(list(set(myCustomDict[key])))

    
    graph = myCustomDict

    # print(graph)

    return graph

def showBasic(bfslist):

    x = []
    p = []

    myData = open("30node.txt", "r").read().split("\n")
    d = bfslist
    lenth_of_bfs = len(bfslist)
    print(lenth_of_bfs)
    
    # p = myData[0].replace("(","").replace(")","").replace("[","").replace("]","").replace("'","") .split(", ")
    

    for item in range(len(myData)):
        x = myData[item].replace("(","").replace(")","").replace("[","").replace("]","").replace("'","").split(", ")

        if(bfslist[0]==x[0]): 
            if(bfslist[1]==x[1]):
                val1 = Node(int(x[3]),int(x[4]))
                val2 = Node(int(x[5]),int(x[6]))
                val3 = val1.distance(val2)


        if(bfslist[1]==x[0]): 
            if(bfslist[2]==x[1]):
                val4 = Node(int(x[3]),int(x[4]))
                val5 = Node(int(x[5]),int(x[6]))
                val6 = val4.distance(val5)

                
                # print(Node(int(x[3]),int(x[4])).get())
                # print(Node(int(x[5]),int(x[6])).get())
            
    # print((val3+val6)/2)
    print(val3)
            # p.append(x)
    
    return d.pop(0)

# # Unit tests
# x=Node(4,5)
# y=Node(1,2)
# print(x.get())
# print(y.get())
# print(x.distance(y))



# main searcher class
class Searcher():
    myViz = DRDViz() #using this to save graph with path to new .png file
    l = []
    open = []
    
    
    
    def __init__(self,filepath):
        self.filepath = filepath
        self.open = open
        x.loadGraphFromFile(filepath)  # load it up from a map file
        x.plot()  # make the plot of the road map appear
        pass
    
    
    


    def setStartGoal(self,a,b):
        self.a = a  #start node
        self.b = b  #goal node
        
        x.markStart(a)  # mark start node
        x.markGoal(b)   # mark goal node 
        
        # self.l = bfs_shortest_path(loadData(self.filepath),self.a,self.b)
        self.l = bfs(loadData(self.filepath),self.a,self.b)
        self.open = bfs(loadData(self.filepath),self.a,self.b)
        showBasic(self.open)
        print(bfs(loadData(self.filepath),self.a,self.b))
        x.paintPath(self.l,color='r') #paint the path

        return self.l

    
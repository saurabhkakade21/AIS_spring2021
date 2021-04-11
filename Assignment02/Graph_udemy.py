#   Created by Elshad Karimov 
#   Copyright © 2021 AppMillers. All rights reserved.

class Graph:
    def __init__(self, gdict=None):
        if gdict is None:
            gdict = {}
        self.gdict = gdict
    
    def addEdge(self, vertex, edge):
        self.gdict[vertex].append(edge)
    
    def bfs(self, vertex):
        visited = [vertex]
        queue = [vertex]
        while queue:
            deVertex = queue.pop(0)
            node = deVertex[-1]
            
            if node not in visited:
                adjacentVertex = vertex[node]
            # print(deVertex)
                for adjacentVertex in self.gdict[deVertex]:
                    visited.append(adjacentVertex)
                    queue.append(adjacentVertex)
    
    def dfs(self, vertex):
        visited = [vertex]
        stack = [vertex]
        while stack:
            popVertex = stack.pop()
            print(popVertex)
            for adjacentVertex in self.gdict[popVertex]:
                if adjacentVertex not in visited:
                    visited.append(adjacentVertex)
                    stack.append(adjacentVertex)
    

def loadData():

    myData = open("30node.txt", "r").read().split("\n")

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

        # secondList = []

    # print(myCustomDict)

    # thirdList = []

    # for i in range(len(myThirdData)):

    #     currSel = myThirdData[i][1]
    #     # l = list()
    #     # # print(myThirdData[i][4].split(", ["))
    #     # print(''.join(map(str,myThirdData[i][4])))

    #     # for i in range(len(myThirdData)):
    #     #     l.append(myThirdData[i][4].split(","))

    #     # for j in range(len(mySecondData)):
    #     #     myThirdData.append(myData[j].rsplit("'"))
        

    #     for j in range(len(myThirdData)):

    #         if(myThirdData[j][3] == currSel):

    #             thirdList.append(myThirdData[j][1])

    #     # myCustomDict[currSel].append(thirdList)

    #     for x in range(len(thirdList)):

    #         myCustomDict[currSel].append(thirdList[x])

    #     thirdList = []

    #     # print(myCustomDict)


    #     for key in myCustomDict.items():

    #         print(list(set(myCustomDict[key])))

    #         # myCustomDict[key] = sorted(list(set(myCustomDict[key])))

    
    graph = myCustomDict

    return graph

abc = loadData()

# customDict = { "a" : ["b","c"],
#             "b" : ["a", "d", "e"],
#             "c" : ["a", "e"],
#             "d" : ["b", "e", "f"],
#             "e" : ["d", "f", "c"],
#             "f" : ["d", "e"]
#                }



g = Graph(abc)
g.dfs("N")


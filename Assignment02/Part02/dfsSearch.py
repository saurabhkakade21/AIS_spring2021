# This class represent a graph
class Graph:
    # Initialize the class
    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
    # Create an undirected graph by adding symmetric edges
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist
    # Add a link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance
    # Get neighbors or a neighbor
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)
    # Return a list of nodes in the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)
# This class represent a node
class Node:
    # Initialize the class
    def __init__(self, name:str, parent:str):
        self.name = name
        self.parent = parent
        self.g = 0 # Distance to start node
        self.h = 0 # Distance to goal node
        self.f = 0 # Total cost
    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name
    # Sort nodes
    def __lt__(self, other):
         return self.f < other.f
    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))
# Depth-first search (DFS)
def dfs(graph, start, end):
    
    # Create lists for open nodes and closed nodes
    openList = []
    closedList = []
    # Create a start node and an goal node
    startNode = Node(start, None)
    goalNode = Node(end, None)
    # Add the start node
    openList.append(startNode)
    
    # Loop until the openList is empty
    while len(openList) > 0:
        # Get the last node (LIFO)
        currentNode = openList.pop(-1)
        # Add the current node to the closedList
        closedList.append(currentNode)
        
        # Check if we have reached the goal, return the path
        if currentNode == goalNode:
            path = []
            while currentNode != startNode:
                path.append(currentNode.name + ': ' + str(currentNode.g))
                currentNode = currentNode.parent
            path.append(startNode.name + ': ' + str(startNode.g))
            # Return reversed path
            return path[::-1]
        # Get neighbours
        neighbors = graph.get(currentNode.name)
        # Loop neighbors
        for key, value in neighbors.items():
            # Create a neighbor node
            neighbor = Node(key, currentNode)
            # Check if the neighbor is in the closedList
            if(neighbor in closedList):
                continue
            # Check if neighbor is in openList and if it has a lower f value
            if(neighbor in openList):
                continue
            # Calculate cost so far
            neighbor.g = currentNode.g + graph.get(currentNode.name, neighbor.name)
            # Everything is green, add neighbor to openList
            openList.append(neighbor)
    # Return None, no path is found
    return None
# The main entry point for this module
def main():
    # Create a graph
    graph = Graph()
    # Create graph connections (Actual distance)
    # graph.connect('N', 'AB', 299)
    # graph.connect('U', 'H', 72)
    # graph.connect('V', 'U', 185)
    # graph.connect('S', 'N', 69)
    # graph.connect('N', 'AA', 100)
    # graph.connect('AA', 'S', 52)
    # graph.connect('AA', 'B', 91)
    # graph.connect('K', 'M', 138)
    # graph.connect('H', 'K', 41)
    # graph.connect('K', 'U', 78)
    # graph.connect('U', 'C', 64)
    # graph.connect('W', 'A', 84)
    # graph.connect('V', 'W', 399)
    # graph.connect('A', 'U', 337)
    # graph.connect('Y', 'N', 166)
    # graph.connect('I', 'Y', 272)
    # graph.connect('I', 'F', 413)
    # graph.connect('Y', 'AA', 62)
    # graph.connect('P', 'F', 201)
    # graph.connect('R', 'P', 176)
    # graph.connect('M', 'P', 85)
    # graph.connect('Q', 'E', 449)
    # graph.connect('AB', 'Q', 477)
    # graph.connect('Q', 'D', 63)
    # graph.connect('D', 'E', 231)
    # graph.connect('N', 'Q', 192)
    # graph.connect('Q', 'S', 126)
    # graph.connect('S', 'D', 102)
    # graph.connect('AD', 'V', 599)
    # graph.connect('AC', 'E', 227)
    # graph.connect('D', 'AC', 134)
    # graph.connect('S', 'AC', 131)
    # graph.connect('B', 'AC', 119)
    # graph.connect('B', 'AE', 48)
    # graph.connect('AE', 'W', 84)
    # graph.connect('K', 'X', 77)
    # graph.connect('C', 'X', 119)
    # graph.connect('X', 'F', 196)
    # graph.connect('AC', 'O', 108)
    # graph.connect('O', 'E', 219)
    # graph.connect('C', 'J', 54)
    # graph.connect('J', 'F', 64)
    # graph.connect('AE', 'J', 266)
    # graph.connect('U', 'J', 178)
    # graph.connect('T', 'Y', 65)
    # graph.connect('F', 'T', 99)
    # graph.connect('J', 'T', 203)
    # graph.connect('AE', 'T', 87)
    # graph.connect('T', 'AA', 93)
    # graph.connect('L', 'AC', 99)
    # graph.connect('W', 'L', 171)
    # graph.connect('L', 'O', 67)
    # graph.connect('L', 'AD', 88)
    # graph.connect('E', 'AD', 100)

    graph.connect('D', 'I', 377)
    graph.connect('I', 'B', 509)
    graph.connect('B', 'D', 275)
    graph.connect('A', 'E', 608)
    graph.connect('G', 'A', 127)
    graph.connect('A', 'H', 72)
    graph.connect('H', 'D', 135)
    graph.connect('J', 'H', 223)
    graph.connect('J', 'F', 127)
    graph.connect('I', 'C', 291)
    graph.connect('C', 'K', 270)
    graph.connect('F', 'C', 278)
    graph.connect('C', 'E', 293)
    graph.connect('L', 'F', 277)
    graph.connect('J', 'L', 193)
    graph.connect('L', 'C', 134)
    graph.connect('G', 'B', 235)
      

    # Make graph undirected, create symmetric connections
    graph.make_undirected()
    # Run search algorithm
    path = dfs(graph, 'L', 'C')
    print(path)
    print()
# Tell python to run main method
if __name__ == "__main__": main()


from program2_funs import Searcher 
# , hSLD, SNode


# (b) Show your program loading in the 30-node sample file. 
s = Searcher("30node.txt")

# (c) Show you program setting start node=U and end node=T. 
s.setStartGoal('U','T')

# myViz should be a DRDViz instance -> save map to file on disk.
s.myViz.save("30node1.png")

# (d) Show the one open node.
# [n.showBasic() for n in s.open]

# for n in s.open:
    
#     print(n)


    
# # (e) Show successors of only open node.
# initial_children = s.successors(s.open.pop(0))

# [n.showBasic() for n in initial_children]


# # (f) Show three inserts: at the front, and the end, and "in order"
# def reset_insert(where):
#     s.reset()
#     initial_children = s.successors(s.open.pop(0))
#     insert_method = getattr(s, "insert_"+where)
#     insert_method(initial_children)
#     return [n.showBasic() for n in s.open]

# reset_insert("front")
# reset_insert("end")
# reset_insert("ordered")

# # (g) INSERT (K,500), (C,91) and (J,10) and show no duplicates.
# newdata = (("K",500), ("C",91), ("J",10))
# newlist = [SNode(label=label, pathcost=pathcost) for label, pathcost in newdata]
# ignored = s.insert_end(newlist)
# [n.showBasic() for n in s.open]

# # 3. hSLD heuritic function being called on three nodes.
# [hSLD(x, s) for x in ("V", "AC", "J")]

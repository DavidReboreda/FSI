# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)
an = search.GPSProblem('A', 'N', search.romania)
on = search.GPSProblem('O', 'N', search.romania)
ob = search.GPSProblem('O', 'B', search.romania)
aa = search.GPSProblem('A', 'A', search.romania)

outfile = open('Busquedas.txt', 'w')

print "Camino A-B: "
outfile.write("Camino A-B:\n")
outfile.write("Busqueda primero en anchura --> ")
outfile.close()
print search.breadth_first_graph_search(ab).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Busqueda primero en profundidad --> ")
outfile.close()
print search.depth_first_graph_search(ab).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion --> ")
outfile.close()
print search.busqueda1(ab).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion con subestimacion --> ")
outfile.close()
print search.busqueda(ab).path()

outfile = open("Busquedas.txt", 'a')
print "Camino A-N: "
outfile.write("Camino A-N:\n")
outfile.write("Busqueda primero en anchura --> ")
outfile.close()
print search.breadth_first_graph_search(an).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Busqueda primero en profundidad --> ")
outfile.close()
print search.depth_first_graph_search(an).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion --> ")
outfile.close()
print search.busqueda1(an).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion con subestimacion --> ")
outfile.close()
print search.busqueda(an).path()

outfile = open("Busquedas.txt", 'a')
print "Camino O-N: "
outfile.write("Camino O-N:\n")
outfile.write("Busqueda primero en anchura --> ")
outfile.close()
print search.breadth_first_graph_search(on).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Busqueda primero en profundidad --> ")
outfile.close()
print search.depth_first_graph_search(on).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion --> ")
outfile.close()
print search.busqueda1(on).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion con subestimacion --> ")
outfile.close()
print search.busqueda(on).path()


outfile = open("Busquedas.txt", 'a')
print "Camino O-B: "
outfile.write("Camino O-B:\n")
outfile.write("Busqueda primero en anchura --> ")
outfile.close()
print search.breadth_first_graph_search(ob).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Busqueda primero en profundidad --> ")
outfile.close()
print search.depth_first_graph_search(ob).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion --> ")
outfile.close()
print search.busqueda1(ob).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion con subestimacion --> ")
outfile.close()
print search.busqueda(ob).path()


outfile = open("Busquedas.txt", 'a')
print "Camino A-A: "
outfile.write("Camino A-A:\n")
outfile.write("Busqueda primero en anchura --> ")
outfile.close()
print search.breadth_first_graph_search(aa).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Busqueda primero en profundidad --> ")
outfile.close()
print search.depth_first_graph_search(aa).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion --> ")
outfile.close()
print search.busqueda1(aa).path()
outfile = open('Busquedas.txt', 'a')
outfile.write("Ramificacion y acotacion con subestimacion --> ")
outfile.close()
print search.busqueda(aa).path()

"""


print "Camino A-N: "
print search.breadth_first_graph_search(an).path()
print search.depth_first_graph_search(an).path()
#print search.iterative_deepening_search(an).path()
#print search.depth_limited_search(an).path()

print "Camino N-B: "
print search.breadth_first_graph_search(on).path()
print search.depth_first_graph_search(on).path()
#print search.iterative_deepening_search(on).path()
#print search.depth_limited_search(on).path()

print "Camino O-B: "
print search.breadth_first_graph_search(ob).path()
print search.depth_first_graph_search(ob).path()
#print search.iterative_deepening_search(ob).path()
#print search.depth_limited_search(ob).path()

print "Camino A-A: "
print search.breadth_first_graph_search(aa).path()
print search.depth_first_graph_search(aa).path()
#print search.iterative_deepening_search(aa).path()
#print search.depth_limited_search(aa).path()



print search.busqueda(an).path()
print search.busqueda1(an).path()

print search.busqueda(on).path()
print search.busqueda1(on).path()

print search.busqueda(ob).path()
print search.busqueda1(ob).path()

print search.busqueda(aa).path()
print search.busqueda1(aa).path()




"""

#print search.busqueda(ab)
#print search.busqueda(ab)[1]
#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450

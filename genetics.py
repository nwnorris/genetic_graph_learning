import random
import math
from igraph import *

#Genetic parameters
init_pop_size = 64
init_genome_size = 3
mutation_chance = 0.1
add_sub_edge_chance = 0.1
change_edge_chance = 1 - add_sub_edge_chance

max_edge_id = 36

class Creature():
    def __init__(self, init_size=None, genome=None):
        if(genome):
            self.genome = genome
        else:
            self.genome = ""
            for x in range(init_size):
                self.genome += "{:0>2d}".format(math.floor(random.random() * max_edge_id))

        self.genes = []
        j = 0
        for x in range(len(self.genome) // 2):
            end = (x+1)*2
            gene = int(self.genome[j:end])
            j = end
            self.genes.append(gene)

        self.size = len(self.genome) // 2
        self.fitness = -1

    def mutate(self):
        if(random.random() <= mutation_chance):
            #Choose what kind of mutation
            if(random.random() <= add_sub_edge_chance):
                if(random.random() > 0.5):
                    #Add an edge
                    to_add = random.randint(0, max_edge_id-1)
                    if(to_add not in self.genes):
                        self.genome += "{:0>2d}".format(to_add)
                        self.genes.append(to_add)
                else:
                    #Delete an edge
                    if(len(self.genome) > 2):
                        self.genome = self.genome[:-2]
                        self.genes = self.genes[:-1]
            else:
                #Change existing edge
                mutate_index = random.randint(0, self.size//2)
                gene = int(self.genes[mutate_index])
                #Current approach: decrement or increment gene.
                #TODO: What about randomly changing?
                direction = 1
                if(random.random() < 0.5 or gene == max_edge_id):
                    direction = -1
                gene += direction
                if(gene < 0):
                    gene = 0
                if(gene not in self.genes):
                    gene = "{:0>2d}".format(gene)
                    self.genome = self.genome[:2*(mutate_index)] + gene + self.genome[2*(mutate_index+1):]
                    self.genes[mutate_index] = int(gene)

    def mate(self, other):

        #print(len(self.genome), len(other.genome))
        crossover_point = 2 * random.randint(0, len(self.genome) // 2)
        child1_genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2_genome = other.genome[:crossover_point] + self.genome[crossover_point:]

        child1 = Creature(genome=child1_genome)
        child2 = Creature(genome=child2_genome)
        child1.mutate()
        child2.mutate()

        # print(self.genome, child1.genome)
        # print(other.genome, child2.genome)
        return [child1, child2]

class Laboratory():
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end
        self.population = []
        self.fitness = [0] * init_pop_size
        self.avgfitness = -1
        for i in range(init_pop_size):
            self.population.append(Creature(init_size=init_genome_size))
        self.calc_fitness()

    def edge_touches_node(self, edge, target):
        try:
            return (target in self.graph.es[edge].tuple)
        except IndexError:
            print(edge)

    def edges_connect_at(self, edge1, edge2, vertex):
        try:
            return (vertex in self.graph.es[edge1].tuple and vertex in self.graph.es[edge2].tuple)
        except IndexError:
            print(edge1, edge2)

    def edges_connected(self, edge1, edge2):
        connect_index = -1
        if(self.edges_connect_at(edge1, edge2, self.graph.es[edge1].source)):
            connect_index = self.graph.es[edge1].source
        elif(self.edges_connect_at(edge1, edge1, self.graph.es[edge1].target)):
            connect_index = self.graph.es[edge1].target

        return connect_index

    def calc_fitness(self):
        avg = 0
        for index, c in enumerate(self.population):

            f = 0
            j = 0
            genes = c.genes
            for i in range(len(genes)):
                #Are any edges connected to start or end?
                connected = set()
                if(self.edge_touches_node(genes[i], start)):
                    f += 10
                    connected.add(start)
                elif(self.edge_touches_node(genes[i], end)):
                    f += 10
                    connected.add(end)

                #Check if i is connected at both ends
                for j in range(len(genes)):
                    conn = self.edges_connected(genes[i], genes[j])
                    if((j != i and conn > -1)):
                        connected.add(conn)
                if(len(connected) == 2):
                    f += 20
                else:
                    f -= 1

                #less fit if longer -- we want optimal solutions
                #f -= int(0.4 * (len(genes)//2))

            self.fitness[index] = f
            c.fitness = f
            avg += f
        self.avgfitness = avg / len(self.population)

    def best_creature(self):
        best = -1
        best_i = -1
        for i, c in enumerate(self.population):
            if(c.fitness > best):
                best = c.fitness
                best_i = i
        return i

    def worst_creature(self):
        worst = -1
        worst_i = -1
        for i, c in enumerate(self.population):
            if(worst == -1 or c.fitness < worst):
                worst = c.fitness
                worst_i = i
        return i

    def best_fitness(self):
        return self.population[self.best_creature()].fitness

    def reproduce(self):
        next_generation = []
        #Kill worst 10% creatures
        for i in range(len(self.population) // 5):
            index = self.worst_creature()
            self.population.pop(index)

        #Pad up to max with best creatures
        while(len(self.population) < init_pop_size):
            self.population.append(Creature(genome=self.population[self.best_creature()].genome))

        #Select pairs & reproduce
        while(len(self.population) > 0):
            parent1 = self.population.pop(random.randint(0, len(self.population)-1))
            parent2 = self.population.pop(random.randint(0, len(self.population)-1))
            for child in parent1.mate(parent2):
                next_generation.append(child)
        self.population = next_generation

    def next_generation(self):
        l.reproduce()
        l.calc_fitness()

def parse_graph(filename):
    file = open(filename, "r").readlines()
    #Split each line by space & convert to int
    result =  list(map(lambda l: tuple(map(lambda i: int(i), l.split(" "))), file))
    g = Graph()
    g.add_vertices(36)
    g.add_edges(result)
    return g

#Driver code
graph = parse_graph("hw3_graph.txt")

start = int(input("Enter start node: "))
end = int(input("Enter end node: "))
print("Breeding population to find route from " + str(start) + " to " + str(end))

l = Laboratory(graph, start, end)

print(l.avgfitness)
best_solution = [-500, -1]
for x in range(25000):
    l.next_generation()
    best = l.population[l.best_creature()]
    if(best.fitness > best_solution[0]):
        best_solution[0] = best.fitness
        best_solution[1] = best.genome
    print(int(l.avgfitness), l.best_fitness(), l.population[0].genome)

print("Best solution found: ", best_solution)

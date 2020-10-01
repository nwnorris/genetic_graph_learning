import random
import math
import pygame
import copy
from igraph import *

#Genetic parameters
init_pop_size = 128
init_genome_size = 1
mutation_chance = 0.08
add_sub_edge_chance = 0.5
change_edge_chance = 1 - add_sub_edge_chance

max_edge_id = 36

class LaboratoryGUI():

    def __init__(self, lab):
        self.laboratory = lab
        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.bg_rect = pygame.Rect(0, 0, self.width, self.height)
        self.base_graph = lab.graph.layout_kamada_kawai()
        self.base_graph.fit_into((0, 0, self.width, self.height))
        self.font = pygame.font.Font(pygame.font.match_font("Arial"), 12)
        self.base_graph.scale(0.9)
        self.edge_lines = self.generate_edge_lines()
        self.vertex_radius = 20

    def generate_edge_lines(self):
        edges = []
        for e in self.laboratory.graph.es:
            start = [int(z) for z in self.base_graph.coords[e.source]]
            end = [int(z) for z in self.base_graph.coords[e.target]]
            print(e.source, e.target)
            edges.append([start, end])
        return edges

    def update(self):

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                running = False

        pygame.draw.rect(self.screen, pygame.Color("#dddddd"), self.bg_rect)

        #Draw edges
        for e in self.edge_lines:
            pygame.draw.line(self.screen, pygame.Color('#000000'), e[0], e[1], 1)

        #Draw best population member
        best = self.laboratory.best_creature()
        for g in best.genes:
            edge = self.edge_lines[g]
            pygame.draw.line(self.screen, [0, 245, 0], edge[0], edge[1], 6)

        #Draw vertices
        for i, c in enumerate(self.base_graph.coords):
            color = [255, 0, 0]
            if(i == self.laboratory.start or i == self.laboratory.end):
                color = [0, 0, 255]
            pygame.draw.circle(self.screen, color, [int(z) for z in c], self.vertex_radius)


            id = self.font.render(str(i), True, [255, 255, 255], color)
            self.screen.blit(id, [int(z - 6) for z in c])

        pygame.display.flip()

class Creature():
    def __init__(self, init_size=None, genome=None):
        self.genes = []
        if(genome):
            self.genome = genome
            j = 0
            for x in range(len(self.genome) // 2):
                end = (x+1)*2
                gene = int(self.genome[j:end])
                j = end
                self.genes.append(gene)
        else:
            self.genome = ""
            for x in range(init_size):
                eid = math.floor(random.random() * max_edge_id)
                if(eid not in self.genes):
                    self.genome += "{:0>2d}".format(eid)
                    self.genes.append(eid)
                else:
                    x -= 1




        self.size = len(self.genome) // 2
        self.fitness = -1
        self.eval = ""

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
                mutate_index = random.randint(0, len(self.genes)//2)
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
        #Chance to mutate again
        if(random.random() <= 0.5):
            self.mutate()

    def mate(self, other, keep=None):

        #print(len(self.genome), len(other.genome))
        crossover_point = 2 * random.randint(0, len(self.genome) // 2)
        child1_genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2_genome = other.genome[:crossover_point] + self.genome[crossover_point:]

        child1 = Creature(genome=child1_genome)
        child2 = Creature(genome=child2_genome)

        if(keep):
            if(keep[0]):
                child1 = self
            if(keep[1]):
                child2 = other

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
        self.last_avg_fitness = 0
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



    def graph_from_gene_set(self, genes):
        edges = []
        for g in genes:
            s = self.graph.es[g].source
            t = self.graph.es[g].target
            edges.append((s, t))

        genome_graph = Graph()
        genome_graph.add_vertices(19)
        genome_graph.add_edges(edges)

        return genome_graph

    #A solution is a subset of the primary graph that connects the start and end vertices
    def is_solution(self, graph):
        #Identify if the genome has the start and end vertex connected to it.
        #Along the way, make sure every vertex that isn't the start or end has degree of 2 (connected inline with rest)
        has_start = False
        has_end = True

        #for g in genes:
        for e in graph.es:
            source = e.source_vertex
            valid_source = ((source == self.start or source== self.end) and source.degree() == 1)
            target = e.target_vertex
            valid_target = ((target == self.end or target == self.start) and target.degree() == 1)
            if(valid_source):
                has_start =  True
                if(target.degree != 1):
                    return False
            elif(valid_target):
                has_end = True
                if(source.degree() != 1):
                    return False
            else:
                #Valid solution must have connected edges throughout! Much like the straight line solution in the fitness calculation.
                if(source.degree() != 1 or target.degree() != 1):
                    return False
        if(has_start and has_end):
            print(genes, " is a solution.")
            return True
        return False

    def calc_fitness(self):
        avg = 0
        for index, c in enumerate(self.population):
            fit_eval = ['-'] * 5
            f = 0
            j = 0
            genes = c.genes
            test_graph = self.graph_from_gene_set(c.genes)

            #Check for connection to start/end
            start = False
            end = False
            for e in test_graph.es:
                if(self.start in e.tuple):
                    start = True
                    f += 40
                    fit_eval[0] = 'S' #Start
                elif(self.end in e.tuple):
                    end = True
                    f += 40
                    fit_eval[1] = 'E' #End

            #Place a value on a "straight line" solution -- all edges are connected and non-looping
            #Another way to describe this situation is that given n vertices, the degree of 2 vertices is 1, and then n-2 vertices have degree 2
            #We know this is true because for any graph, if any edge is "disconnected" from the rest, or connected in a branching capacity, there must be > 2 1-degree vertices
            single_deg_ct = 0
            double_deg_ct = 0
            many_deg_ct = 0
            for e in test_graph.es:
                degs = [e.source_vertex.degree(), e.target_vertex.degree()]
                for d in degs:
                    if d == 1:
                        single_deg_ct += 1
                    elif d == 2:
                        double_deg_ct += 1
                    elif d > 2:
                        many_deg_ct += 1

            #Another heuristic: misconnection of start/end
            #No solution is going to have a start/end vertex with degree > 1
            #Weight heavily negative to seriously push to not have this.
            if(test_graph.vs[self.start].degree() > 1 or test_graph.vs[self.end].degree() > 1):
                fit_eval[4] = 'M' #Misconnected
                f -= 500

            #Is this a valid solution?
            if self.is_solution(test_graph):
                f += 20 * 17 #Solution should be more favorable than any non-solution genome
                #For solutions, shorter is better
                fit_eval[3] = 'V'
                f += int(1000 / len(genes))
            else:
                fit_eval[3] = 'I'
                #For non-solutions, longer is better -- we probably need to add more edges to find a solution
                f += 5 * int((len(genes)//2))

                #Without the increase in fitness depending on the number of connections in the solution, a 2-edge straight line is no better than a 3-edge straight line.
                #And, while we're searching for a solution, longer lines will help us find a path. Usually.
                if(single_deg_ct == 2):
                    fit_eval[2] = 'L' #Line
                    #f += 10 * double_deg_ct
                    f += 20
                else:
                    fit_eval[2] = 'B' #Branching
                    f -= 20 * single_deg_ct

                    #Encourage nodes to not have many connections
                    f -= 60 * many_deg_ct

            c.eval = "".join(fit_eval)
            c.fitness = f
            self.fitness[index] = f
            avg += f
        self.avgfitness = avg / len(self.population)

    def best_creature(self):
        #if(self.best == None):
        best = -5000
        best_i = -1
        for i, c in enumerate(self.population):
            if(c.fitness > best):
                best = c.fitness
                best_i = i
        self.best = self.population[best_i]
        return self.best

    def worst_creature(self):
        worst = 99999
        worst_i = -1
        for i, c in enumerate(self.population):
            if(worst == -1 or c.fitness < worst):
                worst = c.fitness
                worst_i = i
        return i

    def best_fitness(self):
        return self.best_creature().fitness

    def reproduce(self):
        next_generation = []
        #Kill worst 10% creatures
        for i in range(len(self.population) // 5):
            index = self.worst_creature()
            self.population.pop(index)

        #Pad up to max with best creatures
        best = self.best_creature()
        while(len(self.population) < init_pop_size):
            copy = Creature(genome=best.genome)
            copy.fitness = best.fitness
            self.population.append(best)

        #Select pairs & reproduce
        while(len(self.population) > 0):
            keep = [False, False]
            parent1 = self.population.pop(random.randint(0, len(self.population)-1))
            parent2 = self.population.pop(random.randint(0, len(self.population)-1))
            if(parent1.fitness > self.last_avg_fitness * 1.3):
                keep[0] = True
            elif(parent2.fitness > self.last_avg_fitness * 1.3):
                keep[1] = True

            for child in parent1.mate(parent2, keep):
                next_generation.append(child)

        self.population = next_generation

        for g in self.population:
            g.mutate()

    def next_generation(self):
        self.best = None
        self.reproduce()
        self.calc_fitness()
        self.last_avg_fitness = self.avgfitness

def parse_graph(filename):
    file = open(filename, "r").readlines()
    #Split each line by space & convert to int
    result =  list(map(lambda l: tuple(map(lambda i: int(i), l.split(" "))), file))
    g = Graph()
    g.add_vertices(19)
    g.add_edges(result)
    return g

#Driver code
pygame.init()
graph = parse_graph("hw3_graph.txt")

start = int(input("Enter start node: "))
end = int(input("Enter end node: "))
print("Breeding population to find route from " + str(start) + " to " + str(end))

l = Laboratory(graph, start, end)
g = LaboratoryGUI(l)

print(l.avgfitness)
best_solution = [-500, -1]
for x in range(25000):
    l.next_generation()
    best = l.best_creature()
    if(best.fitness > best_solution[0]):
        best_solution[0] = best.fitness
        best_solution[1] = best.genome
    print(int(l.avgfitness), best.fitness, best.genome, best.eval)
    g.update()
print("Best solution found: ", best_solution)
g.update()

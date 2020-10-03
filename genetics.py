import random
import math
import pygame
import copy
from igraph import *

#Genetic parameters
init_pop_size = 128
init_genome_size = 1
mutation_chance = 0.20
add_sub_edge_chance = 0.5
change_edge_chance = 1 - add_sub_edge_chance

max_edge_id = 36

class LaboratoryGUI():

    def __init__(self, graph):
        #self.laboratory = lab
        self.graph = graph
        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.bg_rect = pygame.Rect(0, 0, self.width, self.height)
        self.base_graph = graph.layout_kamada_kawai()
        self.base_graph.fit_into((0, 0, self.width, self.height))
        self.font = pygame.font.Font(pygame.font.match_font("Arial"), 12)
        self.base_graph.scale(0.8)
        self.base_graph.translate((self.width * 0.2) / 2, (self.height * 0.2) / 2)
        self.edge_lines = self.generate_edge_lines()
        self.vertex_radius = 20
        self.graph_previous_ct = 100
        self.graph_previous = []
        self.graph_rect = pygame.Rect(350, 20, self.graph_previous_ct, 50)
        #self.choose_target_vertices()

    def setLaboratory(self, laboratory):
        self.lab = laboratory

    def get_hovered_vertex(self, mouse):
        for i, c in enumerate(self.base_graph.coords):
            dist = math.sqrt((mouse[0] - c[0])**2 + (mouse[1] - c[1])**2)
            if(dist <= self.vertex_radius):
                return i
        return -1

    def choose_target_vertices(self):
        self.target = []

        done = False
        selected = []
        hovered = -1

        while(not done):
            mouse = pygame.mouse.get_pos()
            hovered = self.get_hovered_vertex(mouse)

            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    done = True
                if e.type == pygame.MOUSEBUTTONUP:
                    if(hovered not in selected):
                        selected.append(hovered)
                    else:
                        selected.remove(hovered)
                if e.type == pygame.KEYDOWN:
                    if(e.key == 13 and len(selected) > 0): #Enter key
                        return selected


            self.draw_base_graph(target_vs = selected, outline = [hovered])
            pygame.display.flip()

    def generate_edge_lines(self):
        edges = []
        for e in self.graph.es:
            start = [int(z) for z in self.base_graph.coords[e.source]]
            end = [int(z) for z in self.base_graph.coords[e.target]]
            print(e.source, e.target)
            edges.append([start, end])
        return edges

    def draw_base_graph(self, target_vs=None, best=None, outline=None):
        pygame.draw.rect(self.screen, pygame.Color("#dddddd"), self.bg_rect)

        #Draw edges
        for e in self.edge_lines:
            pygame.draw.line(self.screen, pygame.Color('#000000'), e[0], e[1], 1)

        #Draw best population member
        if(best):
            for g in best.genes:
                edge = self.edge_lines[g]
                pygame.draw.line(self.screen, [0, 245, 0], edge[0], edge[1], 6)

        #Draw vertices
        for i, c in enumerate(self.base_graph.coords):
            color = [255, 0, 0]
            #Color target vertices
            if(target_vs and i in target_vs):
                color = [0, 0, 255]
            pygame.draw.circle(self.screen, color, [int(z) for z in c], self.vertex_radius)
            #Outline mouse-hovered vertices
            if(outline and i in outline):
                pygame.draw.circle(self.screen, [255, 0, 255], [int(z) for z in c], int(self.vertex_radius * 1.2), 2)

            id = self.font.render(str(i), True, [255, 255, 255], color)
            self.screen.blit(id, [int(z - 6) for z in c])

    def draw_info(self):
        font_color = [0, 0, 0]
        gens = self.font.render("Generation " + str(self.lab.generations), True, font_color, pygame.Color("#dddddd"))
        fit = self.font.render("Best fitness: " + str(self.lab.best.fitness), True, font_color, pygame.Color("#dddddd"))
        edges = self.font.render("# Edges: " + str(len(self.lab.best.genome) // 2), True, font_color, pygame.Color("#dddddd"))
        gen = self.font.render("Best genome: " + str(self.lab.best.genome), True, font_color, pygame.Color("#dddddd"))
        info  = [gens, fit, edges, gen]
        pos = [20, 20]
        for i in info:
            self.screen.blit(i, pos)
            pos[1] += 20

    def graph_fitness(self, best):
        self.graph_previous.append(best)
        if(len(self.graph_previous) > self.graph_previous_ct):
            self.graph_previous.pop(0)

        pygame.draw.rect(self.screen, pygame.Color("#383838"), self.graph_rect, 1)
        for i, fit in enumerate(self.graph_previous):
            if(i < len(self.graph_previous) - 1):
                y2 = self.graph_rect.y + self.graph_rect.height - (self.graph_previous[i+1] / 2200 * self.graph_rect.height)
                y1 = self.graph_rect.y + self.graph_rect.height - (fit / 2200 * self.graph_rect.height)
                pygame.draw.line(self.screen, pygame.Color("#383838"), [self.graph_rect.x + i, int(y1)], [self.graph_rect.x + (i + 1), int(y2)], 1)

                if(i == len(self.graph_previous) - 2):
                    xmax = self.font.render(str(self.lab.generations), True, [0, 0, 0], pygame.Color("#dddddd"))
                    self.screen.blit(xmax, [self.graph_rect.x + i, self.graph_rect.y + self.graph_rect.height])
                    fit_text = self.font.render(str(int(fit)), True, [0, 0, 0], pygame.Color("#dddddd"))
                    self.screen.blit(fit_text, [self.graph_rect.x - 23, int(y2)])

    def update(self, target_vs, best):
        self.draw_base_graph(target_vs=target_vs, best=best)
        self.draw_info()
        self.graph_fitness(best.fitness)
        pygame.display.flip()

class Creature():
    def __init__(self, init_size=None, genome=None):
        self.genes = []
        if(genome):
            unique = set()
            j = 0
            for x in range(len(genome) // 2):
                end = (x+1)*2
                gene = (genome[j:end])
                j = end
                if(gene not in unique):
                    self.genes.append(int(gene))
                    unique.add(gene)
            self.genome = "".join(list(unique))

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
                #Current approach: randomly select a new edge to mutate to
                gene = random.randint(0, max_edge_id-1)
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
    def __init__(self, graph, target):
        self.graph = graph
        self.target = target
        self.population = []
        self.fitness = [0] * init_pop_size
        self.avgfitness = -1
        self.last_avg_fitness = 0
        self.generations = 0
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
            #print("edge", g, "s:", s, "t:", t)
            edges.append((s, t))


        genome_graph = Graph()
        genome_graph.add_vertices(19)
        genome_graph.add_edges(edges)
        for v in genome_graph.vs:
            v["id"] = v.index

        genome_graph.vs.select(_degree=0).delete()

        return genome_graph

    #Return all target nodes that are within 'graph'
    def target_vs_within(self, graph):
        out = []
        for v in graph.vs:
            if(v.index in self.target):
                out.append(v.index)
        return out

    #A solution is a subset of the primary graph that connects the start and end vertices
    def is_solution(self, graph):
        #Identify if the genome has the target vertices connected to it.
        target_vs = self.target_vs_within(graph)

        #Since there can be no duplicate nodes in the target set, we know this to be true:
        return (len(target_vs) == len(self.target))

    def calc_fitness(self):
        avg = 0
        for index, c in enumerate(self.population):
            fit_eval = ['-'] * 5
            f = 0
            genes = c.genes
            test_graph = self.graph_from_gene_set(c.genes)


            targets_within = []
            single_deg_ct = 0
            for v in test_graph.vs:
                if(v['id'] in self.target):
                    targets_within.append(v)
                else:
                    single_deg_ct += 1

            target_ct = len(targets_within)
            if(target_ct == 0):
                #Solution is useless with no targets in it
                f -= 500
            else:
                #More target vs within graph == better solution
                f += 50 * target_ct


            #Is this a valid solution?
            if len(targets_within) == len(self.target):
                #Base points for containing the solution nodes.
                f += 100

                #More points for being connected
                connected = test_graph.cohesion()
                if(connected != 0):
                    f += 800
                    #More points for fewer edges
                    f += (100 * len(self.graph.vs) / (len(test_graph.es)))
                    #For connected solutions, encourage no single-degree non-target vertices
                    f -= 10 * single_deg_ct
                else:
                    pass

            else:
                #Not a solution -- encourage growth to find connection
                f += 25 * len(test_graph.es)

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
        self.generations += 1

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

g = LaboratoryGUI(graph)
target = g.choose_target_vertices()
#target = [10, 14, 16, 18, 6, 5, 3, 4]
print("Breeding population to find route including",  str(target))

l = Laboratory(graph, target)
g.setLaboratory(l)

best_solution = [-500, -1]

running = True
for x in range(25000):
    l.next_generation()
    best = l.best_creature()
    if(best.fitness > best_solution[0]):
        best_solution[0] = best.fitness
        best_solution[1] = best.genome
    #print(x, int(l.avgfitness), best.fitness, best.genome, best.eval)
    if(x % 10 == 0):
        e = pygame.event.get()
        g.update(target, best)
print("Best solution found: ", best_solution)
g.update()

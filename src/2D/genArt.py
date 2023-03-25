# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
import sys
import random
from copy import deepcopy
import multiprocessing
from datetime import datetime

import numpy
from PIL import Image, ImageDraw, ImageChops

# -------------------------------------------------------------------------------------------------
# Control Variables
# -------------------------------------------------------------------------------------------------

# Genetic Variables
POP_SIZE = 50
MUTATION_CHANCE = 0.05
ADD_GENE_CHANCE = 0.7
REM_GENE_CHANCE = 0.8
INITIAL_GENES = 50

# How often to output images
GENERATIONS_PER_IMAGE = 100

# How many generations to run for
# Set to -1 if you do not want to stop automatically
GENERATIONS_MAX = 4000

# The type of shape to use for the Genes
# Options: [Circle, Square, Triangle]
GENE_TYPE = "Circle"

# -------------------------------------------------------------------------------------------------
# Reference Image
# -------------------------------------------------------------------------------------------------

try:
    globalTarget = Image.open("reference1.png")
    if globalTarget.mode == "RGBA":
        globalTarget = globalTarget.convert('RGB')
except IOError:
    print("The files reference1.png must be located in the same directory as genArt.py.")
    exit()


# -------------------------------------------------------------------------------------------------
# Point and Color Classes
# -------------------------------------------------------------------------------------------------
class Point:
    """
        A 2D point. Has an X and Y coordinate.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Color:
    """
        A color. Has an R, G and B color value.
    """

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


# -------------------------------------------------------------------------------------------------
# Gene Class and SubClasses
# -------------------------------------------------------------------------------------------------
class Gene:
    """
        A Gene is the part of the Organism that can be mutated.
    """

    def __init__(self, size):
        # Used to know the maximum position values
        self.size = size

        # The color of the Gene in RGB
        self.color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # The depth of the gene determines when it is drawn on the canvas relative to the other Genes
        # Genes with a larger depth are drawn first so that the subsequent Genes can be drawn in front of them
        # We reuse the size of the image in the X dimension to bound the "imaginary" Z dimension
        self.depth = random.randint(0, size[0])

    # abstract method
    def mutate(self):
        pass

    def mutateColor(self):
        r = min(max(0, int(round(random.gauss(self.color.r, 10)))), 255)
        g = min(max(0, int(round(random.gauss(self.color.g, 10)))), 255)
        b = min(max(0, int(round(random.gauss(self.color.b, 10)))), 255)

        self.color = Color(r, g, b)

    def mutateDepth(self):
        self.depth = min(max(0, int(round(random.gauss(self.depth, 3)))), self.size[0])


class Circle(Gene):
    """
        A circle Gene. This Gene has a size, diameter, color, and position on the canvas.
    """

    def __init__(self, size):
        super().__init__(size)

        self.diameter = random.randint(5, 15)
        self.pos = Point(random.randint(0, size[0]), random.randint(0, size[1]))
        self.params = ["diameter", "color", "pos"]

    def mutate(self):
        # Decide which variable of the Gene to apply the mutation
        mutation_type = random.choice(self.params)

        # Mutate the variables
        if mutation_type == "diameter":
            self.diameter = min(max(5, int(round(random.gauss(self.diameter, 2)))), 15)

        elif mutation_type == "color":
            self.mutateColor()

        elif mutation_type == "pos":
            x = min(max(0, int(round(random.gauss(self.pos.x, 3)))), self.size[0])
            y = min(max(0, int(round(random.gauss(self.pos.y, 3)))), self.size[1])
            self.pos = Point(x, y)

            # Depth is a sudo z coordinate for the 2-dimensional Gene so mutating the
            # position should also changes the depth
            self.mutateDepth()


class Square(Gene):
    """
        A square Gene. This Gene has a size, color, diagonal length and the position of it's corners on the canvas.
    """

    def __init__(self, size):

        super().__init__(size)

        # The positions of the top left corner of the square on the canvas
        self.pos1 = Point(random.randint(0, size[0]), random.randint(0, size[1]))

        # The diagonal length of the square
        self.diagonal = random.randint(5, 30)

        # The positions of the bottom right corner of the square on the canvas
        self.pos2 = self.calcPos2()
        self.params = ["pos", "color", "diagonal"]

    # For a square the second position is relative to the first position it's diagonal length
    def calcPos2(self):
        x = self.pos1.x + self.diagonal
        y = self.pos1.y + self.diagonal
        return Point(min(x, self.size[0]), min(y, self.size[1]))

    def mutate(self):
        # Decide which variable of the Gene to apply the mutation
        mutation_type = random.choice(self.params)

        # Mutate the variable
        if mutation_type == "pos":
            x = min(max(0, int(round(random.gauss(self.pos1.x, 3)))), self.size[0])
            y = min(max(0, int(round(random.gauss(self.pos1.y, 3)))), self.size[1])

            self.pos1 = Point(x, y)
            self.pos2 = self.calcPos2()

            # Depth is a sudo z coordinate for the 2-dimensional Gene so mutating the position also changes the depth
            self.mutateDepth()

        elif mutation_type == "color":
            self.mutateColor()

        elif mutation_type == "diagonal":
            self.diagonal = min(max(5, int(round(random.gauss(self.diagonal, 2)))), 30)


class Triangle(Gene):
    """
        A triangle Gene. This Gene has a color and 3 positions on the canvas which together form a triangle.
    """

    def __init__(self, size):

        super().__init__(size)

        self.pos1 = Point(random.randint(0, size[0]), random.randint(0, size[1]))
        # Positions 2 and 3 are initialized relative to position 1 to ensure the triangles do not spawn to large
        self.pos2 = self.calcInitialPos()
        self.pos3 = self.calcInitialPos()

        self.params = ["pos", "color"]

    def calcInitialPos(self):
        x = min(max(0, int(round(random.gauss(self.pos1.x, 10)))), self.size[0])
        y = min(max(0, int(round(random.gauss(self.pos1.y, 10)))), self.size[1])
        return Point(x, y)

    def mutate(self):
        # Decide which variable of the Gene to apply the mutation
        mutation_type = random.choice(self.params)

        # Mutate the variable
        if mutation_type == "pos":
            x1 = min(max(0, int(round(random.gauss(self.pos1.x, 3)))), self.size[0])
            y1 = min(max(0, int(round(random.gauss(self.pos1.y, 3)))), self.size[1])
            self.pos1 = Point(x1, y1)

            x2 = min(max(0, int(round(random.gauss(self.pos2.x, 3)))), self.size[0])
            y2 = min(max(0, int(round(random.gauss(self.pos2.y, 3)))), self.size[1])
            self.pos2 = Point(x2, y2)

            x3 = min(max(0, int(round(random.gauss(self.pos3.x, 3)))), self.size[0])
            y3 = min(max(0, int(round(random.gauss(self.pos3.y, 3)))), self.size[1])
            self.pos3 = Point(x3, y3)

            # Depth is a sudo z coordinate for the 2-dimensional Gene so mutating the position also changes the depth
            self.mutateDepth()

        elif mutation_type == "color":
            self.mutateColor()


# -------------------------------------------------------------------------------------------------
# Organism Class
# -------------------------------------------------------------------------------------------------
class Organism:
    """
        The Organism consists of a collection of Genes that work together in order to produce the image.
    """

    def __init__(self, size, num):
        self.size = size
        self.score = 0

        # Create random Genes up to the number given
        if GENE_TYPE == "Circle":
            self.genes = [Circle(size) for _ in range(num)]
        elif GENE_TYPE == "Square":
            self.genes = [Square(size) for _ in range(num)]
        elif GENE_TYPE == "Triangle":
            self.genes = [Triangle(size) for _ in range(num)]
        else:
            print("The GENE_TYPE is not a valid type.")
            sys.exit()

    def mutate(self):
        # Each Gene has a random chance of mutating
        # This is statistically equivalent and faster than looping through all the Genes
        mutSampleSize = max(0, int(round(random.gauss(int(len(self.genes) * MUTATION_CHANCE), 1))))
        for g in random.sample(self.genes, mutSampleSize):
            g.mutate()

        # An Organism also has a chance to add or remove a random Gene
        if ADD_GENE_CHANCE > random.random():
            if GENE_TYPE == "Circle":
                self.genes.append(Circle(self.size))
            elif GENE_TYPE == "Square":
                self.genes.append(Square(self.size))
            elif GENE_TYPE == "Triangle":
                self.genes.append(Triangle(self.size))
        if len(self.genes) > 0 and REM_GENE_CHANCE > random.random():
            self.genes.remove(random.choice(self.genes))

    def drawImage(self):
        """
            Using PIL, uses the Genes to draw the Organism as a 2-dimensional image.
        """
        image = Image.new("RGB", self.size, (255, 255, 255))
        canvas = ImageDraw.Draw(image)

        # Sort the Genes by their Z coordinate positions so that they can be drawn in the correct order
        # Genes in the 'back' of the image get drawn first so that the Genes in 'front' get drawn on top of them
        sortedGenes = sorted(self.genes, key=lambda x: x.depth, reverse=True)

        if GENE_TYPE == "Circle":
            for g in sortedGenes:
                color = (g.color.r, g.color.g, g.color.b)
                canvas.ellipse([g.pos.x - g.diameter, g.pos.y - g.diameter, g.pos.x + g.diameter, g.pos.y + g.diameter],
                               outline=color, fill=color)

        elif GENE_TYPE == "Square":
            for g in self.genes:
                color = (g.color.r, g.color.g, g.color.b)
                canvas.rectangle([(g.pos1.x, g.pos1.y), (g.pos2.x, g.pos2.y)], outline=color, fill=color)

        elif GENE_TYPE == "Triangle":
            for g in self.genes:
                color = (g.color.r, g.color.g, g.color.b)
                canvas.polygon([(g.pos1.x, g.pos1.y),
                                (g.pos2.x, g.pos2.y),
                                (g.pos3.x, g.pos3.y)],
                               outline=color, fill=color)
        else:
            print("The GENE_TYPE is not a valid type.")
            sys.exit()

        return image


# -------------------------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------------------------
def run(cores):
    """
        Contains the loop that creates and tests new generations.
    """

    # Create results directory in current directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Create log file
    f = open(os.path.join("results", "log.txt"), 'a')
    f.write("Generation,Accuracy,Genes\n")

    # Start the timer to calculate the run time of the algorithm
    startTime = datetime.now()

    # Create the first parent Organism (with random Genes)
    generation = 1
    parent = Organism(globalTarget.size, INITIAL_GENES)

    # Save an image of the initial Organism to the results directory
    parent.drawImage().save(os.path.join("results", "{}.png".format(generation)))

    # Calculate the score of the Organism
    parent.score = calcScore(parent.drawImage(), globalTarget)

    # Calculate initial accuracy
    accuracy = calcAccuracy(parent.score)

    # Setup the multiprocessing pool
    p = multiprocessing.Pool(cores)

    # Infinite loop (until the process is interrupted)
    while True:
        # Print the current score and write it to the log file
        print("Generation {} - R1: {}%".format(generation, accuracy))
        f.write("{},{},{}\n".format(generation, accuracy, len(parent.genes)))

        # Save an image of the current best Organism to the results directory
        if generation % GENERATIONS_PER_IMAGE == 0:
            parent.drawImage().save(os.path.join("results", "{}.png".format(generation)))

        # Stop program once we are on the max generation
        if generation == GENERATIONS_MAX:
            sys.exit()

        # Increment the generation count
        generation += 1

        # Create the new population and add parent to the collection
        # in case all children mutations result in worse scores
        population = [parent]

        # Generate the children in the new population by applying mutations to the parent )rganism
        try:
            newChildren = generatePopulation(parent, POP_SIZE - 1, p)
        except KeyboardInterrupt:
            # Print the final gene count and total run time
            print("Final Gene Count: {}".format(len(parent.genes)))
            print("Total Runtime: {}\n".format(datetime.now() - startTime))
            p.close()
            return

        # Add the new children to the population
        population.extend(newChildren)

        # Sort the population of Organisms by their fitness.
        sortedPopulation = sorted(population, key=lambda x: fitness(x))

        # Get the Organism that minimises the fitness value the most and
        # assign it to be the parent for the next generation
        parent = sortedPopulation[0]

        # Update the accuracy
        accuracy = calcAccuracy(parent.score)


def fitness(o):
    """
        This functions calculates the fitness of the individual Organisms in the population.
        The fitness is the score for the reference image added to the
        number of Genes the Organisms has multiplied by a weight. The number of Genes is included
        in the score to encourage the Organisms to use the fewest amount of Genes possible.
        The best Organism is the one that is able to minimise this value the most.
    """
    return o.score + len(o.genes) * 2


def calcAccuracy(score):
    """
        This functions takes the score values of an Organism and calculates how accurate
        the image produced by the Organism is to the reference images as a percentage.
        The accuracy is calculated by finding the inverse of the percentage difference.
    """

    # Calculate the total number of color values in the target image by calculating the number of pixels and
    # multiplying by 3 for the three RGB color values.
    imageDimensions = globalTarget.size[0] * globalTarget.size[1] * 3

    return 100 - ((score / 255.0 * 100) / imageDimensions)


def calcScore(im1, im2):
    """
        This function is used to determine how close the image produced by the Organism is to the reference image.
        It uses ImageChops and numpy to quickly compute the sum of the differences
        in the pixels between the two images.
    """
    difImg = ImageChops.difference(im1, im2)
    diff = numpy.sum(difImg)
    return diff


def createChild(o):
    """
        Takes a parent Organism and mutates it's Genes to create a child Organism. The child Organism is then
        tested on the reference image to determine its score.
    """
    try:
        child = deepcopy(o)
        child.mutate()
        img = child.drawImage()
        child.score = calcScore(img, globalTarget)
        return child
    except KeyboardInterrupt as e:
        pass


def generatePopulation(o, number, p):
    """
    Uses the multiprocessing module to create a new population of Organisms.
    """
    population = p.map(createChild, [o] * int(number))
    return population


# -------------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Set number of available cores
    cores = max(1, multiprocessing.cpu_count() - 1)

    # run the Genetic Algorithm
    run(cores)

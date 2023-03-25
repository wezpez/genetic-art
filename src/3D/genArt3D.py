# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------
from vpython import *

from datetime import datetime
import os
import random
from copy import deepcopy
import multiprocessing

import numpy
from PIL import Image, ImageDraw, ImageChops

# -------------------------------------------------------------------------------------------------
# Control Variables
# -------------------------------------------------------------------------------------------------

# Genetic Variables
POP_SIZE = 10
MUTATION_CHANCE = 0.1
ADD_GENE_CHANCE = 0.8
REM_GENE_CHANCE = 0.7
INITIAL_GENES = 50

# Toggle for variable mutation rate
VARIABLE_MUT = True

# How often the variable mutation rate decreases by 0.01
VARIABLE_MUT_RATE = 500

# Toggle to set limit to the size the genes can grow
BOUND_GENE_SIZE = True

# How often to output images
GENERATIONS_PER_IMAGE = 100

# How often to 3D render the organism
GENERATIONS_PER_RENDER = 10

# How many generations to run for
# Set to -1 if you do not want to stop automatically
GENERATIONS_MAX = 15000

# Allows us to run and pause the execution of the program
runProgram = True

# -------------------------------------------------------------------------------------------------
# Reference Images
# -------------------------------------------------------------------------------------------------

try:
    # Get the first reference image
    globalTarget1 = Image.open("reference1.png")
    if globalTarget1.mode == "RGBA":
        globalTarget1 = globalTarget1.convert('RGB')
    # Get the second reference image
    globalTarget2 = Image.open("reference2.png")
    if globalTarget2.mode == "RGBA":
        globalTarget2 = globalTarget2.convert('RGB')
except IOError:
    print("Files reference1.png and reference2.png must be located in the same directory as pyArt-2.py.")
    exit()


# -------------------------------------------------------------------------------------------------
# Point and Color Classes
# -------------------------------------------------------------------------------------------------
class Point:
    """
        A 3D point. Has an X, Y and Z coordinate.
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Color:
    """
        A color. Has an R, G and B color value.
    """

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


# -------------------------------------------------------------------------------------------------
# Genetic Classes
# -------------------------------------------------------------------------------------------------
class Gene:
    """
        A Gene is the part of the Organism that can be mutated.
        This Gene has a size, diameter, color, and position in the 3D sculpture.
    """

    def __init__(self, size):
        # Used to know the maximum position values
        self.size = size

        self.diameter = random.randint(5, 15)
        # Reusing the size of the image in the X dimension to bound the "imaginary" Z dimension
        self.pos = Point(random.randint(0, size[0]), random.randint(0, size[1]), random.randint(0, size[0]))
        self.color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.params = ["diameter", "pos", "color"]

    def mutate(self):
        # Decide which variable will be mutated
        mutation_type = random.choice(self.params)

        # Mutate the variable
        if mutation_type == "diameter":
            if BOUND_GENE_SIZE:
                self.diameter = min(max(5, int(round(random.gauss(self.diameter, 2)))), 15)
            else:
                self.diameter = max(5, int(round(random.gauss(self.diameter, 2))))

        elif mutation_type == "pos":
            x = min(max(0, int(round(random.gauss(self.pos.x, 5)))), self.size[0])
            y = min(max(0, int(round(random.gauss(self.pos.y, 5)))), self.size[1])
            z = min(max(0, int(round(random.gauss(self.pos.z, 5)))), self.size[0])
            self.pos = Point(x, y, z)

        elif mutation_type == "color":
            r = min(max(0, int(round(random.gauss(self.color.r, 20)))), 255)
            g = min(max(0, int(round(random.gauss(self.color.g, 20)))), 255)
            b = min(max(0, int(round(random.gauss(self.color.b, 20)))), 255)
            self.color = Color(r, g, b)


class Organism:
    """
        The Organism consists of a collection of Genes that work together in order to produce the images and 3D sculpture.
    """

    def __init__(self, size, num):
        self.size = size
        # Create random Genes up to the number given
        self.genes = [Gene(size) for _ in range(num)]
        # The score is first initialized to (0, 0) and is then calculated later
        self.score = (0, 0)

    def mutate(self):
        # Each Gene has a random chance of mutating
        # This is statistically equivalent and faster than looping through all the Genes
        mutSampleSize = max(0, int(round(random.gauss(int(len(self.genes) * MUTATION_CHANCE), 1))))
        for g in random.sample(self.genes, mutSampleSize):
            g.mutate()

        # The organism also has a chance to add or remove a Gene when it mutates
        if ADD_GENE_CHANCE > random.random():
            self.genes.append(Gene(self.size))
        if len(self.genes) > 0 and REM_GENE_CHANCE > random.random():
            self.genes.remove(random.choice(self.genes))

    def drawImage(self, perspective):
        """
            Using PIL, the Genes draw the Organism as a 2-dimensional image from a specified perspective.
        """
        image = Image.new("RGB", self.size, (255, 255, 255))
        canvas = ImageDraw.Draw(image)

        if perspective == 1:
            # Sort the Genes by their Z coordinate positions so that they can be drawn in the correct order
            sortedGenes = sorted(self.genes, key=lambda x: x.pos.z, reverse=True)

            for g in sortedGenes:
                color = (g.color.r, g.color.g, g.color.b)
                canvas.ellipse([g.pos.x - g.diameter, g.pos.y - g.diameter, g.pos.x + g.diameter, g.pos.y + g.diameter],
                               outline=color, fill=color)
        else:
            # Sort the Genes by their X coordinate positions so that they can be drawn in the correct order
            sortedGenes = sorted(self.genes, key=lambda x: x.pos.x, reverse=True)

            for g in sortedGenes:
                color = (g.color.r, g.color.g, g.color.b)
                canvas.ellipse(
                    [(self.size[0] - g.pos.z) - g.diameter, g.pos.y - g.diameter, (self.size[0] - g.pos.z) + g.diameter,
                     g.pos.y + g.diameter],
                    outline=color, fill=color)

        return image


# -------------------------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------------------------
def run(cores):
    """
        Contains the loop that creates and tests new generations.
    """
    global runProgram

    # Create results directory in current directory
    if not os.path.exists("results"):
        os.mkdir("results")

    # Create log file
    logFile = open(os.path.join("results", "log.txt"), 'a')
    logFile.write("Generation,Acc1,Acc2,Genes\n")

    # Start the timer to calculate the run time of the algorithm
    startTime = datetime.now()

    # Create the parent Organism (with random Genes)
    # Can use size of either target image as they must be the same
    generation = 1
    parent = Organism(globalTarget1.size, INITIAL_GENES)

    # Save an image of the initial Organism
    parent.drawImage(1).save(os.path.join("results", "P1_{}.png".format(generation)))
    parent.drawImage(2).save(os.path.join("results", "P2_{}.png".format(generation)))

    # Calculate the score of the Organism from the different perspectives compared to the two target images
    parent.score = (calcScore(parent.drawImage(1), globalTarget1), calcScore(parent.drawImage(2), globalTarget2))

    # Calculate initial accuracy
    accuracy = calcAccuracy(parent.score)

    # Create the counters for the display
    generationCounter = wtext(pos=display.title_anchor, text="\n<b>Generation:</b> {}\n".format(generation))
    refAccuracy1 = wtext(pos=display.title_anchor, text="<b>Accuracy to reference1:</b> {}%\n".format(accuracy[0]))
    refAccuracy2 = wtext(pos=display.title_anchor, text="<b>Accuracy to reference2:</b> {}%\n".format(accuracy[1]))
    geneCounter = wtext(pos=display.title_anchor, text="<b>Number of Genes:</b> {}\n".format(len(parent.genes)))

    # Create the curves for the Graph
    accuracyCurve1 = gcurve(color=color.blue, width=4, label='Ref. 1')
    accuracyCurve2 = gcurve(color=color.red, width=4, label='Ref. 2')

    # Render the initial Organism in 3D
    renderOrganism(parent)

    # Setup the multiprocessing pool
    p = multiprocessing.Pool(cores)

    # Infinite loop (until the process is interrupted)
    while True:
        # Pause and run the GA
        if runProgram:
            # Print the current score and write generation info to the log file
            print("Generation {} - R1: {}%, R2: {}%".format(generation, round(accuracy[0], 3), round(accuracy[1], 3)))
            logFile.write("{},{},{},{}\n".format(generation, accuracy[0], accuracy[1], len(parent.genes)))

            # Save an image of the current best Organism to the results directory
            if generation % GENERATIONS_PER_IMAGE == 0:
                parent.drawImage(1).save(os.path.join("results", "P1_{}.png".format(generation)))
                parent.drawImage(2).save(os.path.join("results", "P2_{}.png".format(generation)))

            # Update the accuracy
            accuracy = calcAccuracy(parent.score)

            # Update the display counters
            generationCounter.text = "\n<b>Generation:</b> {}\n".format(generation)
            refAccuracy1.text = "<b>Accuracy to reference1:</b> {}%\n".format(accuracy[0])
            refAccuracy2.text = "<b>Accuracy to reference2:</b> {}%\n".format(accuracy[1])
            geneCounter.text = "<b>Number of Genes:</b> {}\n".format(len(parent.genes))

            # Render the current best Organism in 3D
            if generation % GENERATIONS_PER_RENDER == 0:
                renderOrganism(parent)
                # Update the graph
                accuracyCurve1.plot(generation, accuracy[0])
                accuracyCurve2.plot(generation, accuracy[1])

            # Pause program once we are on the max generation
            if generation == GENERATIONS_MAX:
                logFile.write("Total Runtime: {}\n".format(datetime.now() - startTime))
                runProgram = False

            # Decrease mutation rate if using variable mutation
            if VARIABLE_MUT and generation % VARIABLE_MUT_RATE == 0:
                global MUTATION_CHANCE
                if MUTATION_CHANCE > 0.01:
                    MUTATION_CHANCE = MUTATION_CHANCE - 0.01

            # Increment the generation count
            generation += 1

            # Create the new population and add parent to the collection
            # in case all children mutations result in worse scores
            population = [parent]

            # Generate the children in the new population by applying mutations to the parent organism
            try:
                newChildren = generatePopulation(parent, POP_SIZE - 1, p)
            except KeyboardInterrupt:
                # Print the final Gene count and total run time
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


def fitness(o):
    """
        This functions calculates the fitness of the individual Organisms in the population.
        The fitness is the sum of the scores for each of the reference images added to the
        number of Genes the Organisms has multiplied by a weight. The number of Genes is included
        in the score to encourage the Organisms to use the fewest amount of Genes possible.
        The best Organism is the one that is able to minimise this value the most.
    """
    return o.score[0] + o.score[1] + len(o.genes) * 2


def calcAccuracy(score):
    """
        This functions takes the score values of an Organism and calculates how accurate
        the image produced by the Organism is to the reference images as a percentage.
        The accuracy is calculated by finding the inverse of the percentage difference.
    """

    # Calculate the total number of color values in the target images by calculating the number of pixels and
    # multiplying by 3 for the three RGB color values. Only need size of one, images must be same sized
    imageDimensions = globalTarget1.size[0] * globalTarget1.size[1] * 3

    # Since the score is a calculation of the sum of the differences between the two images
    # we are easily able to calculate the % difference and then use the inverse of this for the accuracy
    return (100 - ((score[0] / 255.0 * 100) / imageDimensions)), (100 - ((score[1] / 255.0 * 100) / imageDimensions))


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
        tested on both reference images and given a score.
    """
    try:
        child = deepcopy(o)
        child.mutate()
        img1 = child.drawImage(1)
        score1 = calcScore(img1, globalTarget1)
        img2 = child.drawImage(2)
        score2 = calcScore(img2, globalTarget2)
        child.score = (score1, score2)
        return child
    except KeyboardInterrupt as e:
        pass


def generatePopulation(o, number, p):
    """
        Uses the multiprocessing module to create a new population of Organisms in parallel.
    """
    population = p.map(createChild, [o] * int(number))
    return population


# -------------------------------------------------------------------------------------------------
# Vpython helper methods
# -------------------------------------------------------------------------------------------------

# Toggle the execution of the main program
def runButton(b):
    global runProgram
    if b.text == 'Pause':
        runProgram = False
        b.text = 'Run'
    else:
        runProgram = True
        b.text = 'Pause'


# Deletes all the objects in the scene
def clearScene():
    for obj in display.objects:
        obj.visible = False
        del obj


# Takes an Organism and renders it in Vpython
def renderOrganism(organism):
    # Clear the scene from any previously rendered Organisms
    clearScene()

    # Find the midpoint of the 3D image so that we can subtract it from each
    # point position so that the rendered image is centered around (0,0,0).
    xTranslate = organism.size[0] // 2
    yTranslate = organism.size[1] // 2
    # Reusing xTranslate for zTranslate
    zTranslate = xTranslate

    for g in organism.genes:
        # Reformat Gene attributes to be compatible with Vpython
        p = vec((g.pos.x - xTranslate), (0 - (g.pos.y - yTranslate)), (0 - (g.pos.z - zTranslate)))
        # Doubling the radius because the size of the balls was too small before
        r = g.diameter
        c = vec((g.color.r / 255), (g.color.g / 255), (g.color.b / 255))

        # Create sphere object from Gene
        sphere(pos=p, radius=r, color=c)


# Takes a screen shot of the 3D scene and saves it as a .png in your downloads folder
def takeScreenShot():
    display.capture("PyArt2")


# -------------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set number of available cores
    cores = max(1, multiprocessing.cpu_count() - 1)

    # Create the 3D scene
    display = canvas()
    display.width = display.height = 600
    display.background = color.white
    display.align = 'left'
    display.range = 1

    # Setting the FOV to be small to produce a Orthographic projection
    display.fov = 0.01

    # lighting
    display.lights = []  # gets rid of default lighting (shadows and distant white light)
    display.ambient = color.gray(1)  # adds bright ambient light

    # Add title
    display.title = '<b>PyArt:</b> Using a Genetic Algorithm to create digital art from multiple reference \n' \
                    'images in the style of Pointillism rendered in 3D.'

    # buttons
    button(text='Pause', bind=runButton)
    button(text='Screen Shot', bind=takeScreenShot)

    # Add a Graph to visualize the improvements
    graph(title='Rate of Improvements', xtitle='Generations', ytitle='Accuracy', align='right', fast=False)

    # Add instructions
    display.append_to_caption("""
    <b>Instructions:</b>
    To rotate "camera", drag with right button or Ctrl-drag.
    To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
      On a two-button mouse, middle is left + right.
    To pan left/right and up/down, Shift-drag.
    Touch screen: pinch/extend to zoom, swipe or two-finger rotate.""")

    # run the genetic algorithm
    run(cores)

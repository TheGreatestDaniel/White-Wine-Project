import numpy as np
import pygame
import sys
import page
import backend

# **** pyGame initialization
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CSC 310 Project -- Decision Tree Path Follower")
font = pygame.font.Font(None, 36)

# Initalize Dataset
generator = backend.generator(
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/winequality-white.csv")

# Create pages
pages = []
treeIndex = 0
current_page_index = 0


def makePage(title, question, pageNum, totalPages, options):
    pages.append(page.Page(title, question, (pageNum, totalPages), screen, font, options))


def makeQuestionPage(index):
    nodeIndex, featureName, threshold, leftChild, rightChild = generator.getNode(index)
    if featureName == "Leaf Node":
        tree = generator.best_dt.tree_
        class_distribution = tree.value[treeIndex][0]

        predicted_quality = np.argmax(class_distribution)
        pages.append(page.Page(f"Predicted Quality: {generator.classifications[predicted_quality]}", screen, font))
        #pages.append(page.Page(str(generator.best_dt.tree_.value[treeIndex]), screen, font))

    pages.append(page.Page("Question " + str(current_page_index + 1), str(featureName),(index, int(generator.best_dt.tree_.node_count)), screen, font, ["<=" + str(round(threshold, 2)), ">" + str(round(threshold, 2))]))


makePage("CSC 310 Project", "White Wine Dataset Visualization", -1, -1, ["Begin"])
makeQuestionPage(0)

decisions = []

# * Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            result = pages[current_page_index].handle_event(event)
            if result != "":
                if "<" in result:
                    treeIndex = generator.getLeft(treeIndex)
                    makeQuestionPage(treeIndex)
                elif ">" in result:
                    treeIndex = generator.getRight(treeIndex)
                    makeQuestionPage(treeIndex)

                decisions.append(result)
                current_page_index += 1

                if result == "Restart":
                    decisions = []
                    pages = [pages[0], pages[1]]
                    current_page_index = 0
                    treeIndex = 0

    pages[current_page_index].draw()
    pygame.display.flip()

pygame.quit()
sys.exit()

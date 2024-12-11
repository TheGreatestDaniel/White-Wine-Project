import pandas as pd
import pygame
import sys
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

class Page:
    def __init__(self, *args):
        if len(args) == 7:  # Constructor for title, question, num, screen, font, button_names, and node_info
            title, question, num, screen, font, button_names, node_info = args
            self.title = title
            self.question = question
            self.screen = screen
            self.font = font
            self.button_names = button_names
            self.buttons = self._create_buttons(self.button_names)
            self.pageNum = num[0]
            self.totalPages = num[1]
            self.node_info = node_info
        elif len(args) == 4:  # Constructor for results, tree image, screen, and font
            results, tree_image_path, screen, font = args
            self.title = "Results"
            self.question = ""
            self.screen = screen
            self.font = font
            self.button_names = ["Restart"]
            self.buttons = self._create_buttons(self.button_names)
            self.tree_image_path = tree_image_path
            self.pageNum = -1
            self.totalPages = -1
            self.node_info = None
        else:
            raise ValueError("Invalid number of arguments for Page initialization")

    def _create_buttons(self, button_names):
        buttons = []
        button_width, button_height = 100, 50
        spacing = 20
        total_width = len(button_names) * button_width + (len(button_names) - 1) * spacing
        start_x = (self.screen.get_width() - total_width) // 2
        y_position = self.screen.get_height() // 2

        for i, name in enumerate(button_names):
            x_position = start_x + i * (button_width + spacing)
            rect = pygame.Rect((x_position, y_position), (button_width, button_height))
            text = self.font.render(name, True, (0, 0, 0))
            buttons.append((rect, text))
        return buttons

    def draw(self):
        LIGHT_BLUE = (173, 216, 230)
        BLACK = (0, 0, 0)
        self.screen.fill(LIGHT_BLUE)

        if self.pageNum != -1:
            text_top_left = self.font.render(f"{self.pageNum}/{self.totalPages}", True, BLACK)
            self.screen.blit(text_top_left, (10, 10))
        
        text_top_middle = self.font.render(self.title, True, BLACK)
        text_below_top_middle = self.font.render(self.node_info or self.question, True, BLACK)

        self.screen.blit(
            text_top_middle,
            (self.screen.get_width() // 2 - text_top_middle.get_width() // 2, 10),
        )
        self.screen.blit(
            text_below_top_middle,
            (self.screen.get_width() // 2 - text_below_top_middle.get_width() // 2, 50),
        )

        if self.title == "Results" and self.tree_image_path:
            try:
                tree_image = pygame.image.load(self.tree_image_path)
                tree_image = pygame.transform.scale(tree_image, (600, 400))
                self.screen.blit(
                    tree_image,
                    (self.screen.get_width() // 2 - tree_image.get_width() // 2, 100),
                )
            except pygame.error:
                error_text = self.font.render("Error loading tree image", True, BLACK)
                self.screen.blit(
                    error_text,
                    (self.screen.get_width() // 2 - error_text.get_width() // 2, 200),
                )

        mouse_pos = pygame.mouse.get_pos()

        for rect, text in self.buttons:
            is_hovered = rect.collidepoint(mouse_pos)
            self._draw_button(rect, text, is_hovered)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, (rect, _) in enumerate(self.buttons):
                if rect.collidepoint(event.pos):
                    return self.button_names[i]
        return ""

    def _draw_button(self, rect, text, is_hovered):
        GRAY = (200, 200, 200)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        color = GRAY if is_hovered else WHITE
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)  # Border
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)


def save_decision_tree_image(model, feature_names, output_path="tree.png"):
    """Generates and saves a decision tree visualization."""
    # Convert class names to strings
    class_names = [str(cls) for cls in model.classes_]

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig(output_path)
    plt.close()


def driver():
    # Load a dataset (Wine Quality dataset as an example)
    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/winequality-white.csv'
    data = pd.read_csv(url)
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # Use DataFrame columns as feature names
    feature_names = X.columns.tolist()

    # Stratified sampling to create a subset
    subset_fraction = 0.2  # Adjust this to control the size of the subset
    X_subset, _, y_subset, _ = train_test_split(X, y, test_size=(1 - subset_fraction), stratify=y, random_state=42)

    # Train a decision tree classifier on the subset
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    # Run the visualization with the subset
    run_visualization(tree_model, feature_names)

# **** pyGame initialization
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Decision Tree Visualization")
font = pygame.font.Font(None, 36)

totalPages = 10
current_page_index = 0
decisions = []

def run_visualization(tree_model, feature_names):
    global current_page_index, decisions

    # Extract tree structure
    tree = tree_model.tree_
    class_names = [str(cls) for cls in tree_model.classes_]
    node_info = [
        f"Node {i}: Feature {feature_names[feature]} <= {threshold:.2f}" if feature != -2 else f"Leaf Node {i}: Predicted Class: {class_names[value.argmax()]}"
        for i, (feature, threshold, value) in enumerate(zip(tree.feature, tree.threshold, tree.value))
    ]

    # Save the tree visualization
    tree_image_path = "tree.png"
    save_decision_tree_image(tree_model, feature_names, tree_image_path)

    # Create pages
    pages = []
    pages.append(Page("Decision Tree Project", "Visualization Example", (-1, -1), screen, font, ["Begin"], None))
    for i in range(len(node_info)):
        pages.append(Page(f"Decision Tree Node {i}", f"Question {i}?", (i, len(node_info)), screen, font, ["Yes", "No"], node_info[i]))
    predicted_value = class_names[tree.value.argmax(axis=2)[0][0]]  # Extract the predicted value at the final node
    pages.append(Page(f"Final Decision: {predicted_value}", None, (-1, -1), screen, font, ["Restart"], None))

    # * Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                result = pages[current_page_index].handle_event(event)
                if result:
                    if result == "Yes" or result == "No":
                        decisions.append(result)
                    current_page_index = min(current_page_index + 1, len(pages) - 1)
                if len(decisions) == len(node_info):
                    current_page_index = len(pages) - 1  # Go to results page
                if result == "Restart":
                    decisions = []
                    current_page_index = 0

        pages[current_page_index].draw()
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    driver()

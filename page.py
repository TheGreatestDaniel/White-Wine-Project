import pygame
class Page:
    def __init__(self, *args):
        if len(args) == 6:  # Constructor for title, question, num, screen, and font
            title, question, num, screen, font, button_names = args
            self.title = title
            self.question = question
            self.screen = screen
            self.font = font
            self.button_names = button_names
            self.buttons = self._create_buttons(self.button_names)
            self.pageNum = num[0]
            self.totalPages = num[1]
        elif len(args) == 3:  # Constructor for results, screen, and font
            results, screen, font = args
            self.title = "You Reached"
            self.question = results
            self.screen = screen
            self.font = font
            self.button_names = ["Restart"]
            self.buttons = self._create_buttons(self.button_names)
            self.pageNum = -1
            self.totalPages = -1
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

        if (self.pageNum != -1):
            text_top_left = self.font.render(str(self.pageNum) + "/" + str(self.totalPages), True, BLACK)
        text_top_middle = self.font.render(self.title, True, BLACK)
        text_below_top_middle = self.font.render(self.question, True, BLACK)

        if (self.pageNum != -1):
            self.screen.blit(text_top_left, (10, 10))
        self.screen.blit(
            text_top_middle,
            (self.screen.get_width() // 2 - text_top_middle.get_width() // 2, 10),
        )
        self.screen.blit(
            text_below_top_middle,
            (self.screen.get_width() // 2 - text_below_top_middle.get_width() // 2, 50),
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
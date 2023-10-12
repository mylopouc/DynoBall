""" Create an example game with pygame"""

import pygame
import sys
import math


def run_pygame():
    # Initialize pygame
    pygame.init()

    # Constants
    width, height = 800, 600
    ball_radius = 20
    section_color = (255, 255, 255)
    box_color = (0, 0, 0)
    ball_color = (255, 0, 0)
    box_width = width//2
    box_height = height//2

    # Initialize font
    font = pygame.font.Font(None, 36)

    # Input box properties
    input_rect_1 = pygame.Rect(20, 0.05*height, 140, 32)
    input_rect_2 = pygame.Rect(20, 0.15*height, 140, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color_1 = color_inactive
    color_2 = color_inactive
    active_1 = False
    active_2 = False
    text_1 = ''
    text_2 = ''
    text_surface_1 = font.render(text_1, True, color_1)
    text_surface_2 = font.render(text_2, True, color_2)

    # Create the game window
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DynoBall")

    # Ball properties
    ball_x = width // 2
    ball_y = height // 2
    box_x = (width // 4)
    box_y = height // 4
    ball_speed = 0
    ball_angle = math.pi / 4  # Initial angle in radians

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_rect_1.collidepoint(event.pos):
                    active_1 = not active_1
                    color_1 = color_active if active_1 else color_inactive
                if input_rect_2.collidepoint(event.pos):
                    active_2 = not active_2
                    color_2 = color_active if active_2 else color_inactive
            if event.type == pygame.KEYDOWN:
                if active_1:
                    if event.key == pygame.K_RETURN:
                        print(f"Input 1: {text_1}")
                        text_1 = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text_1 = text_1[:-1]
                    else:
                        text_1 += event.unicode
                    text_surface_1 = font.render(text_1, True, color_1)
                if active_2:
                    if event.key == pygame.K_RETURN:
                        print(f"Input 2: {text_2}")
                        text_2 = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text_2 = text_2[:-1]
                    else:
                        text_2 += event.unicode
                    text_surface_2 = font.render(text_2, True, color_2)

        # Handle user input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            # ball_speed += 1
            ball_y -= 1
        if keys[pygame.K_DOWN]:
            # ball_speed -= 1
            ball_y += 1
        if keys[pygame.K_LEFT]:
            # ball_angle -= 0.1
            ball_x -= 1
        if keys[pygame.K_RIGHT]:
            ball_angle += 0.1
            ball_x += 1

        # Update ball position
        #ball_x += ball_speed * math.cos(ball_angle)
        #ball_y -= ball_speed * math.sin(ball_angle)

        # Clear the screen
        screen.fill(section_color)

        # Draw the ball
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)

        # Draw rectangular

        # pygame.draw.rect(screen, box_color, (int(box_x), int(box_y), box_width, box_height))
        pygame.draw.rect(screen, box_color, (box_x, box_y, box_width, box_height), 2)

        # Update the display

        width_1 = max(200, text_surface_1.get_width() + 10)
        width_2 = max(200, text_surface_2.get_width() + 10)
        input_rect_1.w = width_1
        input_rect_2.w = width_2
        screen.blit(text_surface_1, (input_rect_1.x + 5, input_rect_1.y + 5))
        screen.blit(text_surface_2, (input_rect_2.x + 5, input_rect_2.y + 5))
        pygame.draw.rect(screen, color_1, input_rect_1, 2)
        pygame.draw.rect(screen, color_2, input_rect_2, 2)

        pygame.display.flip()

        # Limit the frame rate
        pygame.time.Clock().tick(60)

    # Clean up
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    print('Running pygame')
    run_pygame()

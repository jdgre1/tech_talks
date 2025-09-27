import pygame
import random
import math
import numpy as np

# Simulation settings
WIDTH, HEIGHT = 800, 600
NUM_BOIDS = 30
MAX_SPEED = 4
NEIGHBOR_RADIUS = 50
SEPARATION_RADIUS = 20

# Rule weights
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 0.01
SEPARATION_WEIGHT = 1.5

pygame.init()

# Force software rendering to avoid GLX/OpenGL errors
screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.SWSURFACE | pygame.DOUBLEBUF  # Software surface + double buffering
)

pygame.display.set_caption("Boids Simulation with Adjustable Cohesion")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

class Boid:
    def __init__(self):
        self.pos = np.array([random.uniform(0, WIDTH), random.uniform(0, HEIGHT)])
        angle = random.uniform(0, 2 * math.pi)
        self.vel = np.array([math.cos(angle), math.sin(angle)]) * MAX_SPEED

    def update(self, boids):
        neighbors = [b for b in boids if b is not self and np.linalg.norm(b.pos - self.pos) < NEIGHBOR_RADIUS]

        alignment = self.align(neighbors)
        cohesion = self.cohere(neighbors)
        separation = self.separate(neighbors)

        self.vel += (
            ALIGNMENT_WEIGHT * alignment +
            COHESION_WEIGHT * cohesion +
            SEPARATION_WEIGHT * separation
        )

        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = (self.vel / speed) * MAX_SPEED

        self.pos += self.vel
        self.pos[0] %= WIDTH
        self.pos[1] %= HEIGHT

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 255), self.pos.astype(int), 3)

    def align(self, neighbors):
        if not neighbors: return np.zeros(2)
        avg_vel = np.mean([b.vel for b in neighbors], axis=0)
        return avg_vel - self.vel

    def cohere(self, neighbors):
        if not neighbors: return np.zeros(2)
        avg_pos = np.mean([b.pos for b in neighbors], axis=0)
        return (avg_pos - self.pos) * 0.01  # ← cohesion influence

    def separate(self, neighbors):
        force = np.zeros(2)
        for b in neighbors:
            diff = self.pos - b.pos
            dist = np.linalg.norm(diff)
            if 0 < dist < SEPARATION_RADIUS:
                force += diff / dist
        return force

boids = [Boid() for _ in range(NUM_BOIDS)]

running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        COHESION_WEIGHT = min(0.1, COHESION_WEIGHT + 0.001)
    if keys[pygame.K_DOWN]:
        COHESION_WEIGHT = max(0.0, COHESION_WEIGHT - 0.001)

    for boid in boids:
        boid.update(boids)
        boid.draw(screen)

    text = font.render(f'Cohesion Weight: {COHESION_WEIGHT:.3f} (↑/↓)', True, (255, 255, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

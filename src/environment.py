import numpy as np

class LaneKeepingEnv:
    def __init__(self, road_width=10, lane_width=3):
        self.road_width = road_width
        self.lane_width = lane_width
        self.position = road_width // 2  # Start in the center of the road
        self.done = False
        self.total_reward = 0

    def reset(self):
        self.position = self.road_width // 2
        self.done = False
        self.total_reward = 0
        return self.position

    def step(self, action):
        # Actions: 0 = left, 1 = stay, 2 = right
        if action == 0:
            self.position -= 1
        elif action == 2:
            self.position += 1

        # Reward system
        reward = 1 if abs(self.position - self.road_width // 2) <= self.lane_width else -10
        self.total_reward += reward

        # Check if the car crashes (out of road)
        if self.position < 0 or self.position > self.road_width:
            self.done = True

        return self.position, reward, self.done

    def render(self):
        road = ['-' for _ in range(self.road_width)]
        if 0 <= self.position < self.road_width:
            road[self.position] = 'C'  # Car position
        print(''.join(road))

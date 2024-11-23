from environment import LaneKeepingEnv
from agent import QLearningAgent

def train_agent(episodes=100):
    env = LaneKeepingEnv()
    agent = QLearningAgent(n_states=env.road_width, n_actions=3)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    train_agent()

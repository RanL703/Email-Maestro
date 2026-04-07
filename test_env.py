from src.executive_assistant.env import AssistantEnv
from src.executive_assistant.agent import BaselineAgent  # or whatever name exists

env = AssistantEnv()
agent = BaselineAgent()

obs = env.reset()

for i in range(5):
    action = agent.act(obs)
    obs, reward = env.step(action)
    print(obs)
    print(reward)
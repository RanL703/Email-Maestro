from src.executive_assistant.env import ExecutiveAssistantEnv
from src.executive_assistant.agent import BaselineAgent

# Create env
env = ExecutiveAssistantEnv(task_name="easy_deadline_extraction")

# Create agent
agent = BaselineAgent()

# Reset env
obs = env.reset()

print("STARTING...\n")

# Run loop
for step in range(10):
    decision = agent.choose_action(env.task_name, obs)

    print(f"\nSTEP {step+1}")
    print("Reasoning:", decision.reasoning)
    print("Action:", decision.action)

    obs, reward = env.step(decision.action)

    print("Reward:", reward)
    
    if reward.is_done:
        print("\nTASK COMPLETE ✅")
        break
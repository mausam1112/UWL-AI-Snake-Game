import time
import torch
from src.agent import Agent
from src.game import SnakeGameAI

def play():
    game = SnakeGameAI()
    agent = Agent()

    # Load trained model
    agent.model.load_state_dict(torch.load("models/model-2/model.pth"))
    agent.model.eval()

    while True:
        # Get current state
        state_old = agent.get_state(game)

        # Predict action
        final_move = agent.get_action_from_model(state_old)

        # Perform move
        reward, done, score = game.play_step(final_move)

        if done:
            print("Game over. Score:", score)
            time.sleep(1)
            game.reset()

if __name__ == "__main__":
    play()

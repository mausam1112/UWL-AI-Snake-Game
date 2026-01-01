import argparse
import os
from configs import configs as cfg
from src.game import SnakeGameAI
from src.agent import Agent
from src.helper import plot, get_model_suffix


parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from checkpoint"
)
args = parser.parse_args()

RESUME = args.resume


def train():
    suffix = get_model_suffix()

    game = SnakeGameAI()
    agent = Agent()

    if RESUME:
        try:
            model_folder_path = f"{cfg.MODEL_DIR}/{cfg.MODEL_NAME}-{suffix - 1}"
            
            agent.load_checkpoint(model_folder_path)
            print("Resuming training…")
        except (FileNotFoundError, NotADirectoryError):
            print("No checkpoint found, training from scratch.")
            model_folder_path = f"{cfg.MODEL_DIR}/{cfg.MODEL_NAME}-{suffix}"
    else:
        model_folder_path = f"{cfg.MODEL_DIR}/{cfg.MODEL_NAME}-{suffix}"

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score, 'Highest Score:', agent.highest_score)

            if score > agent.highest_score:
                # storing best score as record 
                agent.highest_score = score

                # saving best model
                agent.model.save(model_folder_path)
                agent.save_checkpoint(model_folder_path)

            agent.scores.append(score)
            agent.total_score += score
            agent.mean_scores.append(agent.total_score / agent.n_games)
            
            plot(
                scores=agent.scores, 
                mean_scores=agent.mean_scores, 
                loss=agent.trainer.track_loss, 
                model_folder_path=model_folder_path, 
                filename="metrics.png"
            )
            # plot(
            #     scores=agent.scores, 
            #     mean_scores=agent.mean_scores, 
            #     model_folder_path=model_folder_path, 
            #     filename="score.png"
            # )
            # plot(
            #     loss=agent.trainer.track_loss, 
            #     model_folder_path=model_folder_path, 
            #     filename="loss.png"
            # )
            


if __name__ == '__main__':
    train()
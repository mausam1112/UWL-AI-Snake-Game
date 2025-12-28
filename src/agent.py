import os
import torch
import random
import numpy as np

from collections import deque
from src.game import Move, Point
from src.model import Linear_QNet, QTrainer
from configs import configs as cfg


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=cfg.MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 3)
        self.trainer = QTrainer(self.model, lr=cfg.LR, gamma=self.gamma)
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.highest_score = 0

    def save_checkpoint(self, model_folder_path, file_name='checkpoint.pt'):
        mem_checkpoint = list(self.memory)
        convert = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        mem_checkpoint = [tuple(map(convert, t)) for t in mem_checkpoint]

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'memory': mem_checkpoint,
            "scores" : self.scores,
            "mean_scores" : self.mean_scores,
            "total_score" : self.total_score,
            "highest_score": self.highest_score,
            "MAX_MEMORY" : cfg.MAX_MEMORY,
            "BATCH_SIZE" : cfg.BATCH_SIZE,
            "LR" : cfg.LR
        }

        filepath = os.path.join(model_folder_path, file_name)
        torch.save(checkpoint, filepath)
        print("Checkpoint saved:", filepath)
        
        # mem_checkpoint = list(self.memory)
        # print(f"{mem_checkpoint[-1] = }")
        # print(f"{type(mem_checkpoint[-1]) = }")

        # convert = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        # mem_checkpoint = [tuple(map(convert, t)) for t in mem_checkpoint]

        # filepath = os.path.join(model_folder_path, "mem_" + file_name)
        # torch.save(mem_checkpoint, filepath)
        # print("Memory saved:", filepath)

    def load_checkpoint(self, model_folder_path, file_name='checkpoint.pt'):
        filepath = os.path.join(model_folder_path, file_name)
        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint['model_state'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.n_games = checkpoint.get('n_games', 0)
        self.epsilon = checkpoint.get('epsilon', 80)
        mem_checkpoint = checkpoint.get('memory')
        self.scores = checkpoint.get('scores', [])
        self.mean_scores = checkpoint.get('mean_scores', [])
        self.total_score = checkpoint.get("total_score", 0)
        self.highest_score = checkpoint.get("highest_score", 0)
        print("Checkpoint loaded:", file_name)

        # filepath = os.path.join(model_folder_path, "mem_" + file_name)
        # mem_checkpoint = torch.load(filepath)
        mem_checkpoint = [
            (np.array(s), a, r, np.array(ns), d ) 
            for (s, a, r, ns, d) in mem_checkpoint
        ]
        self.memory = deque(mem_checkpoint, maxlen=cfg.MAX_MEMORY)
        print("Memory loaded:", filepath)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.move == Move.LEFT
        dir_r = game.move == Move.RIGHT
        dir_u = game.move == Move.UP
        dir_d = game.move == Move.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move Move
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food is in left
            game.food.x > game.head.x,  # food is in right
            game.food.y < game.head.y,  # food is in up
            game.food.y > game.head.y  # food is in down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > cfg.BATCH_SIZE:
            mini_sample = random.sample(self.memory, cfg.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # with increasning game iteration decrease exporation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item()) # argmax returns index which is always integer
            final_move[move] = 1

        return final_move
    
    def get_action_from_model(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        move = [0, 0, 0]
        move[int(torch.argmax(prediction).item())] = 1
        
        return move

    


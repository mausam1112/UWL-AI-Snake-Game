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
        self.MAX_MEMORY = cfg.MAX_MEMORY
        self.BATCH_SIZE = cfg.BATCH_SIZE
        self.LR = cfg.LR

        self.memory = deque(maxlen=self.MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 3)
        self.trainer = QTrainer(self.model, lr=self.LR, gamma=self.gamma)

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
            "MAX_MEMORY" : self.MAX_MEMORY,
            "BATCH_SIZE" : self.BATCH_SIZE,
            "LR" : self.LR,
            'track_loss' : self.trainer.track_loss
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
        self.MAX_MEMORY = checkpoint.get("MAX_MEMORY", cfg.MAX_MEMORY)
        self.BATCH_SIZE = checkpoint.get("BATCH_SIZE", cfg.BATCH_SIZE)
        self.LR = checkpoint.get("LR", cfg.LR)
        self.trainer.track_loss = checkpoint.get("track_loss", [])
        print("Checkpoint loaded:", file_name)

        # filepath = os.path.join(model_folder_path, "mem_" + file_name)
        # mem_checkpoint = torch.load(filepath)
        mem_checkpoint = [
            (np.array(s), a, r, np.array(ns), d ) 
            for (s, a, r, ns, d) in mem_checkpoint
        ]
        self.memory = deque(mem_checkpoint, maxlen=self.MAX_MEMORY)
        print("Memory loaded:", filepath)

    def _distance_until_collision(self, game, start_point, direction):
        """Returns normalized distance until collision in given direction."""
        distance = 0
        point = Point(start_point.x, start_point.y)
        while not game.is_collision(point):
            point = Point(point.x + direction.x, point.y + direction.y)
            distance += 1
        # Normalize by grid width (number of cells)
        return distance / (game.w // cfg.BLOCK_SIZE)

    def _fast_flood_fill(self, grid, start, max_limit=200):
        """Stack-based flood-fill to count reachable free spaces."""
        rows, cols = grid.shape
        r, c = start

        # Clip start coordinates to valid grid
        r = np.clip(r, 0, rows - 1)
        c = np.clip(c, 0, cols - 1)

        r, c = int(r), int(c)

        if grid[r, c] == 1:
            return 0

        visited = np.zeros_like(grid, dtype=bool)
        stack = [(r, c)]
        visited[r, c] = True
        free_count = 0

        while stack:
            r, c = stack.pop()
            free_count += 1
            if free_count >= max_limit:
                return max_limit

            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                # Clip neighbors to grid
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr, nc] and grid[nr, nc] == 0:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
        return free_count

    # ---------------- GET STATE ----------------
    def get_state(self, game):
        head = game.snake[0]

        # One-step neighbors
        point_l = Point(head.x - cfg.BLOCK_SIZE, head.y)
        point_r = Point(head.x + cfg.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - cfg.BLOCK_SIZE)
        point_d = Point(head.x, head.y + cfg.BLOCK_SIZE)

        # Directions
        dir_l = game.move == Move.LEFT
        dir_r = game.move == Move.RIGHT
        dir_u = game.move == Move.UP
        dir_d = game.move == Move.DOWN

        # # Step vectors
        # step_l = Point(-cfg.BLOCK_SIZE, 0)
        # step_r = Point(cfg.BLOCK_SIZE, 0)
        # step_u = Point(0, -cfg.BLOCK_SIZE)
        # step_d = Point(0, cfg.BLOCK_SIZE)

        # DISTANCES & FREE SPACE
        # Build occupancy grid (0 = free, 1 = body/wall)
        grid_h = game.h // cfg.BLOCK_SIZE
        grid_w = game.w // cfg.BLOCK_SIZE
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for p in game.snake:
            row = int(min(p.y // cfg.BLOCK_SIZE, grid_h - 1))
            col = int(min(p.x // cfg.BLOCK_SIZE, grid_w - 1))
            grid[row, col] = 1

        # Ensure food not counted as obstacle
        food_row = int(min(game.food.y // cfg.BLOCK_SIZE, grid_h - 1))
        food_col = int(min(game.food.x // cfg.BLOCK_SIZE, grid_w - 1))
        grid[food_row, food_col] = 0

        # # Head grid coordinates
        head_row = int(min(head.y // cfg.BLOCK_SIZE, grid_h - 1))
        head_col = int(min(head.x // cfg.BLOCK_SIZE, grid_w - 1))

        # Compute distances + free-space depending on current direction
        if dir_r:
            # dist_straight = self._distance_until_collision(game, head, step_r)
            # dist_left     = self._distance_until_collision(game, head, step_u)
            # dist_right    = self._distance_until_collision(game, head, step_d)

            free_straight = self._fast_flood_fill(grid, (head_row, head_col + 1))
            free_left     = self._fast_flood_fill(grid, (head_row - 1, head_col))
            free_right    = self._fast_flood_fill(grid, (head_row + 1, head_col))

        elif dir_l:
            # dist_straight = self._distance_until_collision(game, head, step_l)
            # dist_left     = self._distance_until_collision(game, head, step_d)
            # dist_right    = self._distance_until_collision(game, head, step_u)

            free_straight = self._fast_flood_fill(grid, (head_row, head_col - 1))
            free_left     = self._fast_flood_fill(grid, (head_row + 1, head_col))
            free_right    = self._fast_flood_fill(grid, (head_row - 1, head_col))

        elif dir_u:
            # dist_straight = self._distance_until_collision(game, head, step_u)
            # dist_left     = self._distance_until_collision(game, head, step_l)
            # dist_right    = self._distance_until_collision(game, head, step_r)

            free_straight = self._fast_flood_fill(grid, (head_row - 1, head_col))
            free_left     = self._fast_flood_fill(grid, (head_row, head_col - 1))
            free_right    = self._fast_flood_fill(grid, (head_row, head_col + 1))

        elif dir_d:
            # dist_straight = self._distance_until_collision(game, head, step_d)
            # dist_left     = self._distance_until_collision(game, head, step_r)
            # dist_right    = self._distance_until_collision(game, head, step_l)

            free_straight = self._fast_flood_fill(grid, (head_row + 1, head_col))
            free_left     = self._fast_flood_fill(grid, (head_row, head_col + 1))
            free_right    = self._fast_flood_fill(grid, (head_row, head_col - 1))

        # Normalize free-space
        free_straight /= (grid_h * grid_w)
        free_left     /= (grid_h * grid_w)
        free_right    /= (grid_h * grid_w)

        # ---------------- BUILD STATE VECTOR ----------------
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

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,

            # # Distance-to-obstacle features
            # dist_straight,
            # dist_left,
            # dist_right,

            # Free-space features
            free_straight,
            free_left,
            free_right
        ]

        return np.array(state, dtype=float)

    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
        
    #     dir_l = game.move == Move.LEFT
    #     dir_r = game.move == Move.RIGHT
    #     dir_u = game.move == Move.UP
    #     dir_d = game.move == Move.DOWN

    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or 
    #         (dir_l and game.is_collision(point_l)) or 
    #         (dir_u and game.is_collision(point_u)) or 
    #         (dir_d and game.is_collision(point_d)),

    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or 
    #         (dir_d and game.is_collision(point_l)) or 
    #         (dir_l and game.is_collision(point_u)) or 
    #         (dir_r and game.is_collision(point_d)),

    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or 
    #         (dir_u and game.is_collision(point_l)) or 
    #         (dir_r and game.is_collision(point_u)) or 
    #         (dir_l and game.is_collision(point_d)),
            
    #         # Move Move
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
            
    #         # Food location 
    #         game.food.x < game.head.x,  # food is in left
    #         game.food.x > game.head.x,  # food is in right
    #         game.food.y < game.head.y,  # food is in up
    #         game.food.y > game.head.y  # food is in down
    #     ]

    #     return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, store_loss=True)
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
    
    def get_heuristic_action(self, game):
        """
        Simple heuristic:
        - Prefer moves that get closer to food.
        - Only consider moves that do not cause an immediate collision.
        - If no safe move improves distance, pick any safe move.
        - If no safe move exists, fall back to straight.
        """
        import math
        from src.game import Move, Point

        # Helper: simulate next head position for an action
        def simulate_move(current_move, head, action):
            # [straight, right, left]
            clock_wise = [Move.RIGHT, Move.DOWN, Move.LEFT, Move.UP]
            idx = clock_wise.index(current_move)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                new_dir = clock_wise[(idx + 1) % 4]
            else:  # [0, 0, 1]
                new_dir = clock_wise[(idx - 1) % 4]

            x, y = head.x, head.y
            if new_dir == Move.RIGHT:
                x += cfg.BLOCK_SIZE
            elif new_dir == Move.LEFT:
                x -= cfg.BLOCK_SIZE
            elif new_dir == Move.DOWN:
                y += cfg.BLOCK_SIZE
            elif new_dir == Move.UP:
                y -= cfg.BLOCK_SIZE

            return new_dir, Point(x, y)

        head = game.head
        food = game.food

        possible_actions = [
            np.array([1, 0, 0]),  # straight
            np.array([0, 1, 0]),  # right
            np.array([0, 0, 1])   # left
        ]

        safe_actions = []
        best_action = None
        best_dist = float("inf")

        # Current distance to food
        def dist(p):
            return math.sqrt((p.x - food.x) ** 2 + (p.y - food.y) ** 2)

        for action in possible_actions:
            new_dir, new_head = simulate_move(game.move, head, action)

            # Check immediate collision using game.is_collision
            if game.is_collision(new_head):
                continue  # unsafe move

            safe_actions.append(action)

            d = dist(new_head)
            if d < best_dist:
                best_dist = d
                best_action = action

        # If we found a safe action that improves distance, use it
        def action_to_index(action, possible_actions):
            return next(i for i, a in enumerate(possible_actions) if np.array_equal(a, action))

        if best_action is not None:
            final_move = [0, 0, 0]
            idx = action_to_index(best_action, possible_actions)
            final_move[idx] = 1
            return final_move

        # Otherwise, pick any safe move
        if safe_actions:
            chosen = random.choice(safe_actions)
            final_move = [0, 0, 0]
            idx = possible_actions.index(chosen)
            final_move[idx] = 1
            return final_move

        # If no safe move exists, go straight (we’re doomed anyway)
        return [1, 0, 0]

    


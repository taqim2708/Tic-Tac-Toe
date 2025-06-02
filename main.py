import random
import pickle
import os


class TicTacToe:
    def __init__(self) -> None:
        self.board = [" "] * 9
        self.current_winner: str | None = None

    def available_moves(self) -> list[int]:
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, square: int, letter: str) -> bool:
        if self.board[square] == " ":
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square: int, letter: str) -> bool:
        # Check rows, columns, and diagonals
        row_ind = square // 3
        row = self.board[row_ind * 3 : (row_ind + 1) * 3]
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        col = [self.board[col_ind + i * 3] for i in range(3)]
        if all([s == letter for s in col]):
            return True
        diag1 = [self.board[i] for i in [0, 4, 8]]
        diag2 = [self.board[i] for i in [2, 4, 6]]
        if all([s == letter for s in diag1]) or all([s == letter for s in diag2]):
            return True
        return False

    def is_draw(self) -> bool:
        return " " not in self.board and self.current_winner is None

    def reset(self) -> None:
        self.board = [" "] * 9
        self.current_winner = None

    def get_state(self) -> str:
        return "".join(self.board)

    def print_board(self) -> None:
        for row in [self.board[i * 3 : (i + 1) * 3] for i in range(3)]:
            print("| " + " | ".join(row) + " |")


class QLearningAgent:
    def __init__(
        self,
        letter: str,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        q_table_file: str = "qtable.pkl",
    ) -> None:
        self.q: dict[tuple[str, int], float] = {}
        self.letter = letter
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.load_q_table()

    def load_q_table(self) -> None:
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "rb") as f:
                self.q = pickle.load(f)

    def save_q_table(self) -> None:
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q, f)

    def get_q(self, state: str, action: int) -> float:
        return self.q.get((state, action), 0.5)

    def choose_action(self, game: TicTacToe) -> int:
        state = game.get_state()
        actions = game.available_moves()

        if random.random() < self.epsilon:
            return random.choice(actions)

        qs = [self.get_q(state, a) for a in actions]
        max_q = max(qs)
        max_actions = [a for a, q in zip(actions, qs) if q == max_q]
        return random.choice(max_actions)

    def learn(
        self, state: str, action: int, reward: float, next_state: str, done: bool
    ) -> None:
        old_q = self.get_q(state, action)
        future_q = 0 if done else max([self.get_q(next_state, a) for a in range(9)])
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q[(state, action)] = new_q


def train(n_episodes: int = 50000) -> None:
    game = TicTacToe()
    agent_x = QLearningAgent("X")
    agent_o = QLearningAgent("O")

    for episode in range(n_episodes):
        game.reset()
        agents = {"X": agent_x, "O": agent_o}
        histories = {"X": [], "O": []}
        turn = "X"

        while True:
            agent = agents[turn]
            state = game.get_state()
            action = agent.choose_action(game)
            game.make_move(action, turn)
            next_state = game.get_state()

            # Store the move history
            histories[turn].append((state, action))

            if game.current_winner == turn:
                # Reward all moves made by the winner
                for past_state, past_action in histories[turn]:
                    agent.learn(past_state, past_action, 1, next_state, True)

                # Penalize the opponent for losing early
                total_moves = 9 - len(game.available_moves())
                early_penalty = -1 * (1 + (6 - total_moves) * 0.2)

                loser = "O" if turn == "X" else "X"
                for past_state, past_action in histories[loser]:
                    agents[loser].learn(
                        past_state, past_action, early_penalty, next_state, True
                    )
                break

            elif game.is_draw():
                for p in ["X", "O"]:
                    for past_state, past_action in histories[p]:
                        agents[p].learn(past_state, past_action, 0.5, next_state, True)
                break

            else:
                agent.learn(state, action, 0, next_state, False)
                turn = "O" if turn == "X" else "X"

        if episode % 5000 == 0:
            print(f"Episode {episode / 1000}k")

    agent_x.save_q_table()
    agent_o.save_q_table()
    print("Training complete!")


def play_human() -> None:
    game = TicTacToe()
    ai = QLearningAgent("O")
    game.print_board()

    state = game.get_state()
    last_ai_state = None
    last_ai_action = None

    while True:
        try:
            move = int(input("Your move (0-8): "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if game.make_move(move, "X"):
            next_state = game.get_state()

            if game.current_winner:
                print("You win!")
                if last_ai_state is not None and last_ai_action is not None:
                    ai.learn(last_ai_state, last_ai_action, -1, next_state, True)
                break
            elif game.is_draw():
                print("It's a draw!")
                if last_ai_state is not None and last_ai_action is not None:
                    ai.learn(last_ai_state, last_ai_action, 0.5, next_state, True)
                break

            # AI move
            ai_state = game.get_state()
            ai_action = ai.choose_action(game)
            game.make_move(ai_action, "O")
            game.print_board()
            next_state = game.get_state()

            # Learn from the last move (reward = 0 for now)
            if last_ai_state is not None and last_ai_action is not None:
                ai.learn(last_ai_state, last_ai_action, 0, ai_state, False)

            last_ai_state = ai_state
            last_ai_action = ai_action

            if game.current_winner:
                print("AI wins!")
                ai.learn(last_ai_state, last_ai_action, 1, next_state, True)
                break
            elif game.is_draw():
                print("It's a draw!")
                ai.learn(last_ai_state, last_ai_action, 0.5, next_state, True)
                break
        else:
            print("Invalid move.")

    ai.save_q_table()


if __name__ == "__main__":
    # Step 1: Train the agent
    train(100000)

    agent_o = QLearningAgent("0")
    agent_o.load_q_table()
    print("Agent 0 length", len(agent_o.q))

    agent_x = QLearningAgent("X")
    agent_x.load_q_table()
    print("Agent X length", len(agent_x.q))

    # Step 2: Let human play vs AI
    play_human()

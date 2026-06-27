import random

# Choices
choices = ["rock", "paper", "scissors"]

# Scoreboard
user_score = 0
computer_score = 0
draws = 0

# Store user's move history
user_history = []


# AI Prediction Function
def predict_user_move():
    if len(user_history) < 2:
        # Not enough data, choose randomly
        return random.choice(choices)

    # Count frequency of user's moves
    move_count = {
        "rock": user_history.count("rock"),
        "paper": user_history.count("paper"),
        "scissors": user_history.count("scissors")
    }

    # Predict the move user plays most often
    predicted = max(move_count, key=move_count.get)
    return predicted


# Computer chooses the winning move
def computer_move():
    predicted = predict_user_move()

    if predicted == "rock":
        return "paper"

    elif predicted == "paper":
        return "scissors"

    else:
        return "rock"


# Winner Logic
def find_winner(user, computer):
    global user_score, computer_score, draws

    if user == computer:
        draws += 1
        return "Draw"

    elif (
        (user == "rock" and computer == "scissors") or
        (user == "paper" and computer == "rock") or
        (user == "scissors" and computer == "paper")
    ):
        user_score += 1
        return "You Win!"

    else:
        computer_score += 1
        return "Computer Wins!"


print("=" * 40)
print(" AI Rock Paper Scissors ")
print("=" * 40)

while True:

    user = input("\nEnter Rock, Paper or Scissors (or quit): ").lower()

    if user == "quit":
        break

    if user not in choices:
        print("Invalid Choice!")
        continue

    # Save user's move
    user_history.append(user)

    # AI chooses move
    computer = computer_move()

    print("You      :", user)
    print("Computer :", computer)

    result = find_winner(user, computer)

    print("\nResult :", result)

    print("\nScoreboard")
    print("You      :", user_score)
    print("Computer :", computer_score)
    print("Draws    :", draws)

print("\nFinal Scores")
print("You      :", user_score)
print("Computer :", computer_score)
print("Draws    :", draws)

print("\nThanks for Playing!")
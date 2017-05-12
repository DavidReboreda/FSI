import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

list_actions = {0: "UP",
                1: "RIGHT",
                2: "DOWN",
                3: "LEFT"
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state):
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

def getMaxAction(state):
    if max(Q[state]) > 0:
        index = np.argmax(Q[state])
        return list_actions[index]
    else:
        return getRndAction(state)

    index = np.argmax(Q[state])
    if (Q[state][index] == 0):
        return getRndAction(state)
    else:
        return actions_list.keys()[actions_list.values().index(index)]

def getEGreedyAction(state, chance):
    rnd = random.random()
    if (rnd > chance):
        return getMaxAction(state)
    else:
        return getRndAction(state)

# Episodes
def episodes (strategy, chance):
    global Q
    Q = np.zeros((height * width, num_actions))  # Q matrix
    count = 0.0
    for i in xrange(100):
        state = getRndState()

        while state != final_state:
            if strategy == 0:
                action = getRndAction(state)
            elif strategy == 1:
                action = getMaxAction(state)
            elif strategy == 2:
                action = getEGreedyAction(state, chance)

            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            state = new_state
            count += 1
    return count/100
outfile = open("qlearning.txt", 'w')
print "Default: ", episodes(0, 0)
outfile.write("Promedio de episodios hasta llegar a la meta solo con exploracion: " + str(episodes(0, 0)) + "\n")
print "Greedy: ", episodes(1, 0)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia Greedy: " + str(episodes(1, 0))+ "\n")
print "EGreedy (20%): ", episodes(2, 0.2)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(20%): " + str(episodes(2, 0.2))+ "\n")
print "EGreedy (40%): ", episodes(2, 0.4)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(40%): " + str(episodes(2, 0.4))+ "\n")
print "EGreedy (60%): ", episodes(2, 0.6)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(60%): " + str(episodes(2, 0.6))+ "\n")
print "EGreedy (80%): ", episodes(2, 0.8)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(80%): " + str(episodes(2, 0.8))+ "\n")
print "EGreedy (90%): ", episodes(2, 0.9)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(90%): " + str(episodes(2, 0.9))+ "\n")
print "EGreedy (95%): ", episodes(2, 0.95)
outfile.write("Promedio de episodios hasta llegar a la meta con la estrategia EGreedy(95%): " + str(episodes(2, 0.95))+ "\n")

outfile.close()
#print Q




# Q matrix plot

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()

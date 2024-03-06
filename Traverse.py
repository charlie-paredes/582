import numpy as np
def Viterbi(A,B,pi,O):
    
    num_of_Obs = O.shape[0]
    num_of_States = A.shape[0]

    #path matrix
    psi = np.zeros((num_of_Obs, num_of_States))
    psi.fill(-1.0)

    delta = np.zeros((num_of_Obs, num_of_States))

    best_path = np.zeros(num_of_Obs, dtype=int)

    #initialize viterbi score
    current_score = 0.0

    # Initialization step
    delta[0, :] = pi * B[:, O[0]]

    #Induction Step
    currentRow = np.zeros(num_of_States)

    for t in range(1, num_of_Obs):
        for i in range(num_of_States):
            currentRow.fill(0)
            for j in range(num_of_States):
              currentRow[j] = delta[t-1, j] * A[j, i]

            currMax = np.amax(currentRow)
            delta[t, i] = currMax*B[i, O[t]]
            psi[t, i] = np.argmax(currentRow)

    # Termination Step
    lastRow = delta[num_of_Obs-1]
    current_score = np.amax(lastRow)
    best_path[num_of_Obs-1] = np.argmax(lastRow)

    #Backtracking Step
    # T: The number of states in the best path
    T = best_path.shape[0]

    for t in range(T-2, 0, -1):
        best_path[t] = psi[t+1, best_path[t+1]]

    return current_score, best_path


def main():
    Observation = np.array((0, 1, 1))
    A_prob=np.array(((0.2, 0.8), (0.6, 0.4)))
    B_prob = np.array(((0.7, 0.3), (0.4, 0.6)))
    initial = np.array((1.0, 0.0))
    score, best_path = Viterbi(A_prob, B_prob, initial, Observation)
    print("Viterbi Score: ")
    print(score)
    print("Best Path")
    print(best_path)
main()
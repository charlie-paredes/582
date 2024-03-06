import numpy as np
def Forward(A,B,pi,O):
    
    #Initialization Step
    # Initialize alpha to zero
    alpha=np.zeros((O.shape[0], A.shape[0]))
    alpha[0,:] = np.multiply(pi, B[:,O[0]])
    
    #Induction Step
    for t in range(1, O.shape[0]):
        for j in range(A.shape[0]):
            alphaSum=0.0
            for i in range(A.shape[0]):
                alphaSum += alpha[t-1, i] * A[i, j]

            alpha[t, j] = np.multiply(alphaSum, B[j, O[t]])
    
    #Termination Step
    forwardProb=0
    forwardProb = np.sum(alpha[O.shape[0]-1,:])
    return forwardProb

def main():
    # 0 == H, 1 == T
    Observation = np.array((0, 1, 1))
    A_prob=np.array(((0.2, 0.8), (0.6, 0.4)))
    B_prob = np.array(((0.7, 0.3), (0.4, 0.6)))
    initial = np.array((1.0, 0.0))
    forward = Forward(A_prob, B_prob, initial, Observation)
    print("Result with given 3 Observation vector: ")
    print(f'P(O|\u03BB)={forward:1.4f}')

    Observation = np.array((0, 0, 1))
    forward = Forward(A_prob, B_prob, initial, Observation)
    print("Result with custom 3 Observation vector: ")
    print(f'P(O|\u03BB)={forward:1.4f}')

    Obs_7 = np.array((0,1,1,0,0,1,1))
    forward = Forward(A_prob, B_prob, initial, Obs_7)
    print("Result with 7 Observation vector: ")
    print(f'P(O|\u03BB)={forward:1.4f}')

main()

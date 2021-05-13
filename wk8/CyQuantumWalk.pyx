'''
Cython tutorial - Building a Quantum Walk Simulator

'''
from math import sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef double H[2][2]
H[0][:] = [1/(2**(1/2)),1/(2**(1/2))]
H[1][:] = [1/(2**(1/2)),-1/(2**(1/2))]



cdef ApplyCoinOperator(list spinStates):

    cdef int nSpinStates =  len(spinStates)
    cdef int i
    cdef int j
    cdef int k
    for i in range(<int>nSpinStates):
        for j in range(2):
            for k in range(2):

                spinStates[<int>i][<int>k] += H[<int>j][<int>k]*spinStates[<int>i][<int>k]

    return spinStates

cdef ApplyShiftOperator(list nodeStates, list spinStates):
    cdef int i
    cdef int numberOfNodes = len(nodeStates)
    cdef nodeStatesUpdate = numberOfNodes*[0.0]
    cdef spinStatesUpdate = numberOfNodes*[[0.0, 0.0]]

    for i in range(<int>numberOfNodes):

        if spinStates[<int>i][0] != 0:
            if i != 0:
                nodeStatesUpdate[i - 1] += nodeStates[<int>i]
                spinStatesUpdate[i - 1][0] += spinStates[<int>i][0]
            else:
                nodeStatesUpdate[-1] += nodeStates[<int>i]
                spinStatesUpdate[-1][0] += spinStates[<int>i][0]

        if spinStates[<int>i][1] != 0:
            if i != <int>(numberOfNodes - 1):
                nodeStatesUpdate[i + 1] += nodeStates[<int>i]
                spinStatesUpdate[i + 1][1] += spinStates[<int>i][1]
            else:
                nodeStatesUpdate[0] += nodeStates[<int>i]
                spinStatesUpdate[0][1] += spinStates[<int>i][1]

    nodeStates[:] = nodeStatesUpdate
    spinStates[:] = spinStatesUpdate

    return nodeStates, spinStates

cdef NodeProbabilities(list nodeStates):

    cdef int numberOfNodes = len(nodeStates)
    cdef int i
    cdef list probabilities = []

    for i in range(<int>numberOfNodes):

        probabilities.append(abs(nodeStates[i])**2)

    return probabilities

cdef NormaliseNodeStates(list nodeStates):

    cdef int numberOfNodes = len(nodeStates)
    cdef list probabilities = NodeProbabilities(nodeStates)
    cdef int i
    cdef double totalProbability = 0.0
    cdef double normalisationFactor


    for i in range(<int>numberOfNodes):
        totalProbability += probabilities[i]

    normalisationFactor = 1/sqrt(totalProbability)

    for i in range(<int>numberOfNodes):
        nodeStates[i] *= normalisationFactor

    return nodeStates


cdef NormaliseSpinStates(spinStates):
    
    cdef int numberOfNodes = len(spinStates)
    cdef int i
    cdef int j
    cdef totalProbability = 0

    for i in range(<int>numberOfNodes):
        for j in range(2):
            totalProbability += abs(spinStates[i][j])**2

    cdef double normalisationFactor = 1/sqrt(totalProbability)

    for i in range(<int>numberOfNodes):
        for j in range(2):
            spinStates[i][j] *= normalisationFactor

    return spinStates

def QuantumWalk(numberOfNodes, startingNode, startingSpin, numberOfSteps, ):

    cdef nodeStates = numberOfNodes*[0.0]
    nodeStates[startingNode] = 1.0
    cdef spinStates = numberOfNodes*[[0.0, 0.0]]
    spinStates[startingNode] = startingSpin
    cdef int step
    for step in range(<int>numberOfSteps):

        spinStates = ApplyCoinOperator(spinStates)

        nodeStates, spinStates = ApplyShiftOperator(nodeStates, spinStates)
        
        nodeStates = NormaliseNodeStates(nodeStates)
        spinStates = NormaliseSpinStates(spinStates)
    return NodeProbabilities(nodeStates)

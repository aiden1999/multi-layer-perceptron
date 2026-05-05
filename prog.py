import math
import numpy
from numpy import random

#Importing training data from text file
trainingFile = open(r"trainingData.txt")
initLine = trainingFile.readline()
numCols = len(initLine.split())
trainingFile.seek(0)
initAllTraining = trainingFile.read()
numRowsTraining = len(initAllTraining.split("\n"))
trainingData = numpy.array([initAllTraining.split()]).reshape(numRowsTraining,numCols)
trainingFile.close()

#Importing validation data from text file
validationFile = open(r"validationData.txt")
initAllValidation = validationFile.read()
numRowsValidation = len(initAllValidation.split("\n"))
validationData = numpy.array([initAllValidation.split()]).reshape(numRowsValidation,numCols)
validationFile.close()

#Importing validation data from test file
testFile = open(r"testData.txt")
initAllTest = testFile.read()
numRowsTest = len(initAllTest.split("\n"))
testData = numpy.array([initAllTest.split()]).reshape(numRowsTest,numCols)
testFile.close()

#Create files for RMSE
rmseTrainingFile = open("trainingRMSE.txt","w")
rmseValidationFile = open("validationRMSE.txt","w")
rmseTestFile = open("testRMSE.txt","w")
outputFile = open("output.txt","w")

#Initialising parameters
numNodes = int(input("How many nodes in hidden layer? "))
momentumYN = input("Use momentum? y/n ")
if momentumYN == "y":
    momentum = True
else:
    momentum = False
numInputs = numCols - 1

stepSize = 0.1
correctOutput = 0

weightsInputHidden = numpy.zeros((numNodes, numInputs))
biasHidden = numpy.zeros((numNodes, 1))
weightsHiddenOutput = numpy.zeros((numNodes, 1))
for i in range(numNodes):
    for j in range(numInputs):
        weightsInputHidden[i, j] = random.uniform(-2/numInputs, 2/numInputs)
    biasHidden[i] = random.uniform(-2/numInputs, 2/numInputs)
    weightsHiddenOutput[i] = random.uniform(-2/numNodes, 2/numNodes)
biasOutput = random.uniform(-2/numNodes, 2/numNodes)
sumHidden = numpy.zeros((numNodes, 1))
sumOutput = 0
uHidden = numpy.zeros((numNodes, 1))
uOutput = 0
fPrimeOutput = 0
fPrimeHidden = numpy.zeros((numNodes, 1))
deltaOutput = 0
deltaHidden = numpy.zeros((numNodes, 1))
rmseValidationSum = 0
rmseTestSum = 0
epochCount = 0
repeat = True
rmseValidationOld = 100
alpha = 0.9
weightsInputHiddenOld = numpy.zeros((numNodes, numInputs))
weightsHiddenOutputOld = numpy.zeros((numNodes, 1))
biasHiddenOld = numpy.zeros((numNodes, 1))
biasOutputOld = 0

while repeat:
    epochCount = epochCount + 1
    print(epochCount)
    #Training
    for k in range(numRowsTraining):
        correctOutput = float(trainingData[k, numInputs])
        #Forward Pass
        sumOutput = 0
        for i in range(numNodes):
            sumHidden[i] = 0
            for j in range(numInputs):
                sumHidden[i] = sumHidden[i] + \
                               (float(trainingData[k, j]) * weightsInputHidden[i, j])
            sumHidden[i] = sumHidden[i] + biasHidden[i]
            uHidden[i] = 1 / (1 + numpy.exp(-sumHidden[i]))
            sumOutput = sumOutput + (uHidden[i] * weightsHiddenOutput[i])
        sumOutput = sumOutput + biasOutput
        uOutput = 1 / (1 + numpy.exp(-sumOutput))
        #Backward Pass
        fPrimeOutput = uOutput * (1 - uOutput)
        deltaOutput = (correctOutput - uOutput) * fPrimeOutput
        for i in range(numNodes):
            fPrimeHidden[i] = uHidden[i] * (1 - uHidden[i])
            deltaHidden[i] = weightsHiddenOutput[i] * deltaOutput * fPrimeHidden[i]
        #Update weights and biases
        #With momentum
        if momentum and (epochCount != 1):
            for i in range(numNodes):
                for j in range(numInputs):
                    weightDiff = weightsInputHidden[i,j] - weightsInputHiddenOld[i,j]
                    weightsInputHiddenOld[i,j] = weightsInputHidden[i,j]
                    weightsInputHidden[i,j] = weightsInputHidden[i,j] + \
                                              (stepSize * deltaHidden[i] * \
                                               float(trainingData[k,j])) + \
                                               (alpha * weightDiff)
                weightDiff = weightsHiddenOutput[i] - weightsHiddenOutputOld[i]
                weightsHiddenOutputOld[i] = weightsHiddenOutput[i]
                weightsHiddenOutput[i] = weightsHiddenOutput[i] + \
                                         (stepSize * deltaOutput * uHidden[i]) + \
                                         (alpha * weightDiff)
                biasDiff = biasHidden[i] = biasHiddenOld[i]
                biasHiddenOld[i] = biasHidden[i]
                biasHidden[i] = biasHidden[i] + (stepSize * deltaHidden[i]) + \
                                (alpha * biasDiff)
            biasDiff = biasOutput - biasOutputOld
            biasOutputOld = biasOutput
            biasOutput = biasOutput + (stepSize * deltaOutput) + (alpha * biasDiff)
        #without momentum    
        else:
            for i in range(numNodes):
                for j in range(numInputs):
                    if momentum:
                        weightsInputHiddenOld[i,j] = weightsInputHidden[i,j]
                    weightsInputHidden[i,j] = weightsInputHidden[i,j] + \
                                              (stepSize * deltaHidden[i] * \
                                               float(trainingData[k, j]))
                if momentum:
                    weightsHiddenOutputOld[i] = weightsHiddenOutput[i]
                    biasHiddenOld[i] = biasHidden[i]
                weightsHiddenOutput[i] = weightsHiddenOutput[i] + \
                                         (stepSize * deltaOutput * uHidden[i])
                biasHidden[i] = biasHidden[i] + (stepSize * deltaHidden[i])
            if momentum:
                biasOutputOld = biasOutput
            biasOutput = biasOutput + (stepSize * deltaOutput)
    #RMSE for training data
    rmseTrainingSum = 0
    for k in range(numRowsTraining):
        correctOutput = float(trainingData[k, numInputs])
        sumOutput = 0
        for i in range(numNodes):
            sumHidden[i] = 0
            for j in range(numInputs):
                sumHidden[i] = sumHidden[i] + \
                               (float(trainingData[k, j]) * weightsInputHidden[i, j])
            sumHidden[i] = sumHidden[i] + biasHidden[i]
            uHidden[i] = 1 / (1 + numpy.exp(-sumHidden[i]))
            sumOutput = sumOutput + (uHidden[i] * weightsHiddenOutput[i])
        sumOutput = sumOutput + biasOutput
        uOutput = 1 / (1 + numpy.exp(-sumOutput))
        rmseTrainingSum = rmseTrainingSum + \
                          (((correctOutput - uOutput) ** 2)/numRowsTraining)
    rmseTraining = numpy.sqrt(rmseTrainingSum)
    rmseTrainingFile.write(str(float(rmseTraining)) + "\n")

    #Passing Validation Data every 5 epochs
    if epochCount % 5 == 0:
        rmseValidationSum = 0
        if epochCount != 5:
            rmseValidationOld = rmseValidation
        for k in range(numRowsValidation):
            correctOutput = float(validationData[k, numInputs])
            sumOutput = 0
            for i in range(numNodes):
                sumHidden[i] = 0
                for j in range(numInputs):
                    sumHidden[i] = sumHidden[i] + \
                                   (float(validationData[k, j]) \
                                    * weightsInputHidden[i, j])
                sumHidden[i] = sumHidden[i] + biasHidden[i]
                uHidden[i] = 1 / (1 + numpy.exp(-sumHidden[i]))
                sumOutput = sumOutput + (uHidden[i] * weightsHiddenOutput[i])
            sumOutput = sumOutput + biasOutput
            uOutput = 1 / (1 + numpy.exp(-sumOutput))
            rmseValidationSum = rmseValidationSum + \
                                (((correctOutput - uOutput) ** 2)/numRowsValidation)
        rmseValidation = numpy.sqrt(rmseValidationSum)
        rmseValidationFile.write(str(float(rmseValidation)))
        print("validation " + str(rmseValidation))
        if (rmseValidation > rmseValidationOld) or (epochCount == 10000):
            repeat = False
    rmseValidationFile.write("\n")

#Passing Test Data
for k in range(numRowsTest):
    correctOutput = float(testData[k, numInputs])
    sumOutput = 0
    rmseTestSum = 0
    for i in range(numNodes):
        sumHidden[i] = 0
        for j in range(numInputs):
            sumHidden[i] = sumHidden[i] + \
                           (float(testData[k, j]) * weightsInputHidden[i, j])
        sumHidden[i] = sumHidden[i] + biasHidden[i]
        uHidden[i] = 1 / (1 + math.exp(-sumHidden[i]))
        sumOutput = sumOutput + (uHidden[i] * weightsHiddenOutput[i])
    sumOutput = sumOutput + biasOutput
    uOutput = 1 / (1 + numpy.exp(-sumOutput))
    rmseTestSum = rmseTestSum + (((correctOutput - uOutput) ** 2)/numRowsTest)
    rmseTest = numpy.sqrt(rmseTestSum)
    rmseTestFile.write(str(rmseTest) + "\n")
    outputFile.write(str(float(correctOutput)) + " " + str(float(uOutput)) + "\n")
    print("test " + str(rmseTest))


rmseTrainingFile.close()
rmseValidationFile.close()
rmseTestFile.close()
outputFile.close()

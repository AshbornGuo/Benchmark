import sys
import os
import math
import subprocess
import re

################################
#Change this for a different Case!!!!
desiredVelocity = 1.32
nSlices = 6
zMinCoord = -4
zMaxCoord = 5
yMinCoord = 0
yMaxCoord = 2.9

nBlocks = 18
################################

## Change Fitness for Multiple Instances
USE_FIELD_FITNESS = True

USE_SINGLE_PLANE_FITNESS = False
XY_VELOCITY_FACTOR = 0.1

sliceFitnesses = []

############
## The desired velocity field has to be specified here
## desired Velocity is the average stream velocity
############
def getFunctionValue(z):
    val = -1
    if z>13:
        val = -0.0195*z*z*z+0.7975*z*z-10.893*z+50.85
    else:
        val = 0.000007*z*z*z*z*z-0.00021775*z*z*z*z+0.0019*z*z*z-0.0019*z*z+0.011*z+0.8016
    return val*desiredVelocity

path = "Exeter_CFD_Problems/ESP/foamWorkingDir/"
pathProcessor = path #+ "processor0"
walk = os.walk(pathProcessor)
paths = next(walk)[1]
lastTimeDir = pathProcessor
time = ""
timeBeforeConv = ""
maxTime = 0
maxSecondTime = 0

timeStrings = []

for string in paths:
    if unicode(string).isdecimal():
        timeStrings.append(int(string))
        
timeStrings.sort(reverse=True)

if len(timeStrings) < 2:
    os.system("rm -r " + path)
    sys.exit(-999)
    
time = str(timeStrings[0])
timeBeforeConv = str(timeStrings[1])


#Calculate the fitness of a slice
def getFitness(indexSlice, anaTime = time):
    if USE_FIELD_FITNESS:
        if i == 0:
            filePath = path + "postProcessing/sampleDict/" + anaTime + "/U_slice" + str(i) + ".raw"
            return compareFitnessFieldWithFile(extractFieldForFitness(filePath),"Exeter_CFD_Problems/ESP/fitnessModel")
        else:
            return 0

def extractFieldForFitness(filePath):
    blocks = [[[] for l in range(nBlocks)] for b in range(nBlocks)]
    yDiff = float(float((yMaxCoord - yMinCoord)) / nBlocks)
    zDiff = float(float((zMaxCoord - zMinCoord)) / nBlocks)
    with open(filePath,"r") as rFile:
        nCount = 0
        for line in rFile:
            if nCount > 2:
                splits = line.split()
                if float(splits[2]) > zMinCoord:
                    if float(splits[2]) < zMaxCoord:
                        if float(splits[1]) > yMinCoord and float(splits[1]) < yMaxCoord:
                            y = float(splits[1])
                            z = float(splits[2])
                            blocks[int((y - yMinCoord) / yDiff)][int((z - zMinCoord) / zDiff)].append(line)
            nCount += 1
    fitnessBlocks = [[[] for l in range(nBlocks)] for b in range(nBlocks)]
    for i in range(len(blocks)):
        row = blocks[i]
        for j in range(len(row)):
            block = row[j]
            vel = 0
            for line in block:
                splits = line.split()
                vel += float(splits[3])
            vel /= len(block)
            fitnessBlocks[i][j] = vel
    return fitnessBlocks

def writeFitnessFieldFile(fitnessBlocks,filePath):
    with open(filePath,"w") as rFile:
        for i in range(len(fitnessBlocks)):
            row = fitnessBlocks[i]
            for j in range(len(row)):
                vel = row[j]
                rFile.write(str(i) + " " + str(j) + " " + str(vel) + "\n")

def readFitnessFieldFile(filePath):
    fitnessBlocks = [[[] for l in range(nBlocks)] for b in range(nBlocks)]
    with open(filePath,"r") as rFile:
        for line in rFile:
            if not line.replace(" ","").replace("\n","") == "":
                splits = line.split()
                fitnessBlocks[int(splits[0])][int(splits[1])] = float(splits[2])
    return fitnessBlocks

def compareFitnessFieldWithFile(fitnessBlocks,filePath):
    fitnessBlocksFromFile = readFitnessFieldFile(filePath)
    
    diffs = 0
    for i in range(len(fitnessBlocks)):
        for j in range(len(fitnessBlocks[i])):
            diff = fitnessBlocks[i][j] - fitnessBlocksFromFile[i][j]
            diff = diff*diff
            diffs += diff
    return math.sqrt(diffs/(nBlocks*nBlocks))

logFileDir = path + "log"
foundStr = True

sliceFitnesses = []
fitnessesBeforeConvergence = []

##Request fitness of all slices 
if foundStr:
    ##Sample all timesteps greater than
    for i in range(nSlices):
        sliceFitnesses.append(getFitness(i))
    if not timeBeforeConv == "0":
        for i in range(nSlices):
            fitnessesBeforeConvergence.append(getFitness(i,timeBeforeConv))
    else:
        for i in range(nSlices):
            fitnessesBeforeConvergence.append(0)
else:
    sliceFitnesses.append(-999)
    fitnessesBeforeConvergence.append(0)
    for i in range(nSlices-1):
        sliceFitnesses.append(0)
        fitnessesBeforeConvergence.append(0)

#os.system("rm -r " + path)
sys.exit(sliceFitnesses[0])
#print sliceFitnesses[0]

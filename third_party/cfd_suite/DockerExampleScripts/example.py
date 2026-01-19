import subprocess
import random

## evalFun accepts a vector of integers, length 49 (dimensionality of the ESP problem)
def evalFun(problemName, candidateSolution):
    evalCommand = "docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh " + problemName + " "
    parsedCandidate = ",".join([str(x) for x in candidateSolution])
    return(subprocess.check_output(evalCommand + "'" + parsedCandidate + "'", shell=True))
    
## Create some candidate
candidate = [random.randint(0,7) for i in range(49)]
## Evaluate the candidate, for example on the ESP problem:
print(evalFun("ESP", candidate))

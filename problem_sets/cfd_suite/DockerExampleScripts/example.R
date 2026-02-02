## evalFun accepts a vector of integers, length 49 (dimensionality of the ESP problem)
evalFun <- function(problemName, candidateSolution){
    evalCommand <- paste0("docker run --rm frehbach/cfd-test-problem-suite ./dockerCall.sh ", problemName, " ")
    parsedCandidate <- paste(candidateSolution, sep=",", collapse = ",")
    return(as.numeric(system(paste0(evalCommand, "'", parsedCandidate, "'"), intern = T)))
}

## Create some candidate
candidate <- sample(0:7,49,replace = T)

## Evaluate the candidate, for example on the ESP problem:
evalFun(problemName = "ESP", candidateSolution = candidate)

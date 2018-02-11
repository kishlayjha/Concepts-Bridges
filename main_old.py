import operator
import pathlib
import numpy as np
import class_NearestNeighbour
from class_NearestNeighbour import NearestNeighbour
import os
import sys

def main():
    rootPath = "/home/super-machine/Documents/mydrive/myResearch/output"
    filesPath = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"
    inputTerm1 = "raynaud disease"
    inputTerm2 = "fish oils"
    year = "1985"

    meshTermsInCurrentYear = []
    nearestNeighbour = NearestNeighbour(inputTerm1, year, meshTermsInCurrentYear)
    # nearestNeighbour.loadMeshTermsInGivenYear();
    # embeddingInputTerm1 = nearestNeighbour.getEmbeddings(inputTerm2)
    # print("meshTermsInCurrentYear: ", len(meshTermsInCurrentYear))
    # nearestNeighbour.writeToFileTheNearestNeighbour(embeddingInputTerm1)

    # sys.exit(0)

    startInputTermNearestNeighbor = []
    endInputTermNearestNeighbor = []

    for counter, item in enumerate(open(filesPath+os.sep+inputTerm1+"-sortedByCosine.txt")):
        itemSplit_input = item.split("\t")
        termName_input = itemSplit_input[0].lower()
        startInputTermNearestNeighbor.append(termName_input)
        if(counter>20):
            break;

    for counter, item in enumerate(open(filesPath+os.sep+inputTerm2+"-sortedByCosine.txt")):
        itemSplit_end = item.split("\t")
        termName_end = itemSplit_end[0].lower()
        endInputTermNearestNeighbor.append(termName_end)
        if (counter > 20):
            break;

    print("startInputTermNearestNeighbor: ",len(startInputTermNearestNeighbor))
    print("endInputTermNearestNeighbor: ", len(endInputTermNearestNeighbor))

    allMeshTerm = []
    for counter, item in enumerate(open(filesPath+os.sep+"mesh2017TreeMapping")):
        itemSplit_allMeSH = item.split("\t")
        if (len(itemSplit_allMeSH) > 0):
            meshTermName = itemSplit_allMeSH[0].lower().strip();
            allMeshTerm.append(meshTermName.strip().lower())

    print("allMeshTerm: ",len(allMeshTerm))

    possibleBTerms = []
    for counter, item in enumerate(allMeshTerm):
        if( (item in startInputTermNearestNeighbor) or (item in endInputTermNearestNeighbor)):
            continue
        else:
            possibleBTerms.append(item.lower())

    print("Possible B Terms: ",len(possibleBTerms))

    dictBterms = {}
    for counter, item in enumerate(possibleBTerms):
        print("B term:",item)
        # item = "humans"
        path = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + year + os.sep +item.lower())

        if(path.exists()):
            embedding1 = nearestNeighbour.getEmbeddings(item)
            cosineSimPart1 = 0.0
            for counter1, item1 in enumerate(startInputTermNearestNeighbor):
                # print("start term nearest neighbour: ", item1)
                embedding2 = nearestNeighbour.getEmbeddings(item1)
                cosineSim = nearestNeighbour.cos_sim(embedding1,embedding2)
                if (np.isnan(cosineSim)):
                    continue
                else:
                    cosineSimPart1 = cosineSimPart1+cosineSim

            cosineSimPart2 = 0.0
            for counter2, item2 in enumerate(endInputTermNearestNeighbor):
                # print("end term nearest neighbour: ", item2)
                embeddingPart2 = nearestNeighbour.getEmbeddings(item2)
                cosineSim2 = nearestNeighbour.cos_sim(embedding1, embeddingPart2)
                if (np.isnan(cosineSim)):
                    continue
                else:
                    cosineSimPart2 = cosineSimPart2 + cosineSim2

        if(cosineSimPart1+cosineSimPart2!=0.0):
            finalCosine = 0.5*(cosineSimPart1+cosineSimPart2)
        else:
            finalCosine = 0.0
        # print("cosineSimPart1: ",cosineSimPart1)
        # print("cosineSimPart2: ", cosineSimPart2)
        print(counter,": Final Cosine: ",finalCosine)
        dictBterms[item] = finalCosine
        # sys.exit(0)

    print("Dict B Terms: ",len(dictBterms))

    sorted_BTerms = sorted(dictBterms.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sorted_BTerms:
        meshTerm = key
        count = value
        print("meshTerm: ",meshTerm)
        print("Count: ",count)
        with open(filesPath + os.sep  + "BTermssortedByCosine.txt", 'a') as the_file:
            prepareString = meshTerm + "\t" + str(count)
            the_file.write(prepareString)
            the_file.write("\n")

if __name__ == "__main__":
    main()
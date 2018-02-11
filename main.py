import operator
import pathlib
import numpy as np
import scipy as sp
import scipy.spatial
import class_NearestNeighbour
from class_NearestNeighbour import class_NearestNeighbour
import os
import sys
import settings

settings.init()

def main():
    rootPath = "/home/super-machine/Documents/mydrive/myResearch/output"
    filesPath = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"

    meshTermsInCurrentYear = []
    nearestNeighbour = class_NearestNeighbour(settings.dict['inputTerm1'], settings.dict['year'], meshTermsInCurrentYear)
    # nearestNeighbour.loadMeshTermsInGivenYear();
    # embeddingInputTerm1 = nearestNeighbour.getEmbeddings(inputTerm2)
    # print("meshTermsInCurrentYear: ", len(meshTermsInCurrentYear))
    # nearestNeighbour.writeToFileTheNearestNeighbour(embeddingInputTerm1)

    # sys.exit(0)

    startInputTermNearestNeighbor = []
    endInputTermNearestNeighbor = []

    for counter, item in enumerate(open(filesPath+os.sep+settings.dict['year']+os.sep+settings.dict['inputTerm1']+"-sortedByCosine.txt")):
        itemSplit_input = item.split("\t")
        termName_input = itemSplit_input[0].lower()
        startInputTermNearestNeighbor.append(termName_input)
        if counter> settings.dict['neighboursThreshold']:
            break;

    for counter, item in enumerate(open(filesPath+os.sep+settings.dict['year']+os.sep+settings.dict['inputTerm2']+"-sortedByCosine.txt")):
        itemSplit_end = item.split("\t")
        termName_end = itemSplit_end[0].lower()
        endInputTermNearestNeighbor.append(termName_end)
        if counter > settings.dict['neighboursThreshold']:
            break;

    print("startInputTermNearestNeighbor: ",len(startInputTermNearestNeighbor))
    print("endInputTermNearestNeighbor: ", len(endInputTermNearestNeighbor))

    allMeshTerm = []
    for counter, item in enumerate(open(settings.dict['FILES_PATH']+os.sep+"mesh2017TreeMapping")):
        itemSplit_allMeSH = item.split("\t")
        if (len(itemSplit_allMeSH) > 0):
            meshTermName = itemSplit_allMeSH[0].lower().strip();
            allMeshTerm.append(meshTermName.strip().lower())

    print("allMeshTerm: ",len(allMeshTerm))

    possibleBTerms = []
    for counter, item in enumerate(allMeshTerm):
        if( (item in startInputTermNearestNeighbor) or (item in endInputTermNearestNeighbor) or (item == settings.dict['inputTerm1']) or (item == settings.dict['inputTerm2'])):
            continue
        else:
            possibleBTerms.append(item.lower())

    print("Possible B Terms: ",len(possibleBTerms))


    bTermCosineMatrix = np.empty([len(possibleBTerms), 300])

    for counter, item in enumerate(possibleBTerms):
        # print("B term:",item)
        path = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + settings.dict['year'] + os.sep +item.lower())
        finalCosine = 0.0
        if(path.exists()):
            embedding1 = nearestNeighbour.getEmbeddings(item, settings.dict['year'])
            # print(embedding1)
            bTermCosineMatrix[counter] = embedding1

    print("BTermsMatrix: ",bTermCosineMatrix.shape)

    startTermNearestNeighborMatrix = np.empty([len(startInputTermNearestNeighbor),300])
    for counter2, item2 in enumerate(startInputTermNearestNeighbor):
        path2 = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + settings.dict['year'] + os.sep +item2.lower())

        if(path2.exists()):
            embedding3 = nearestNeighbour.getEmbeddings(item2, settings.dict['year'])
            startTermNearestNeighborMatrix[counter2] = embedding3

    print("startTermNearestNeighborMatrix: ", startTermNearestNeighborMatrix.shape)

    bTermCosinePart1 = 1 - scipy.spatial.distance.cdist(bTermCosineMatrix, startTermNearestNeighborMatrix, 'cosine')
    print("bTermCosinePart1: ",bTermCosinePart1.shape)

    bTermCosinePart1 = bTermCosinePart1.sum(axis=1,dtype=float)
    print("bTermCosinePart1: ",bTermCosinePart1.shape)

    startEndTermNearestNeighborMatrix = np.empty([len(endInputTermNearestNeighbor),300])
    for counter3, item3 in enumerate(endInputTermNearestNeighbor):
        path3 = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + settings.dict['year'] + os.sep +item3.lower())

        if(path3.exists()):
            embedding4 = nearestNeighbour.getEmbeddings(item3, settings.dict['year'])
            startEndTermNearestNeighborMatrix[counter3] = embedding4

    print("startEndInputTermNearestNeighborMatrix: ", startEndTermNearestNeighborMatrix.shape)

    bTermCosinePart2 = 1 - scipy.spatial.distance.cdist(bTermCosineMatrix, startEndTermNearestNeighborMatrix, 'cosine')
    print("bTermCosinePart2: ",bTermCosinePart2.shape)

    bTermCosinePart2 = bTermCosinePart2.sum(axis=1,dtype=float)
    print("bTermCosinePart2: ",bTermCosinePart2.shape)

    bfinalCosine = bTermCosinePart1 + bTermCosinePart2
    print("bTermFinalCosine: ",bfinalCosine.shape)

    dictBterms = {}
    for counter, item in enumerate(possibleBTerms):
        meshTermKey = item
        meshTermVal = bfinalCosine[counter]
        if (np.isnan(meshTermVal)):
            continue
        else:
            dictBterms[meshTermKey] = 0.5*meshTermVal

    sorted_BTerms = sorted(dictBterms.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sorted_BTerms:
        meshTerm = key
        count = value
        with open(filesPath + os.sep+ settings.dict['year']+ os.sep  + "final-sortedBTermsByCosine.txt", 'a') as the_file:
            prepareString = meshTerm + "\t" + str(count)
            the_file.write(prepareString)
            the_file.write("\n")



    # # dictBterms = {}
    # bTermCosineMatrixPart1 = np.empty([len(possibleBTerms),1])
    # # startInputTermNearestNeighborMatrix = np.empty([len(startInputTermNearestNeighbor), 300])
    # for counter, item in enumerate(possibleBTerms):
    #     # print("B term:",item)
    #     path = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + year + os.sep +item.lower())
    #     finalCosine = 0.0
    #     if(path.exists()):
    #         embedding1 = nearestNeighbour.getEmbeddings(item)
    #         # print(embedding1)
    #         for counter1, item1 in enumerate(startInputTermNearestNeighbor):
    #             path1 = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + year + os.sep + item1.lower())
    #
    #             if (path1.exists()):
    #                 embedding2 = nearestNeighbour.getEmbeddings(item1)
    #                 cosine = scipy.spatial.distance.cosine(embedding1, embedding2)
    #                 finalCosine = finalCosine+cosine
    #                 # startInputTermNearestNeighborMatrix[counter1] = embedding2
    #                 bTermCosineMatrixPart1[counter] = finalCosine
    #
    # print("BTermsMatrix: ",bTermCosineMatrixPart1.shape)

    # bTermCosinePart1 = np.dot(bTermsMatrix,startInputTermNearestNeighborMatrix)
    # bTermCosinePart1 = scipy.spatial.distance.cosine(bTermsMatrix, startInputTermNearestNeighborMatrix)
    # bTermCosinePart1 = bTermCosinePart1.sum(axis=1,dtype=float)
    # print("bTermCosinePart1: ",bTermCosinePart1.shape)

    # startEndTermNearestNeighborMatrix = np.empty([len(endInputTermNearestNeighbor),300])
    # for counter2, item2 in enumerate(endInputTermNearestNeighbor):
    #     path2 = pathlib.Path(rootPath + os.sep + "invertedIndex" + os.sep + year + os.sep +item2.lower())
    #
    #     if(path2.exists()):
    #         embedding3 = nearestNeighbour.getEmbeddings(item2)
    #         startEndTermNearestNeighborMatrix[counter2] = embedding3
    #
    # startEndTermNearestNeighborMatrix = startEndTermNearestNeighborMatrix.transpose()
    # print("startEndInputTermNearestNeighborMatrix: ", startEndTermNearestNeighborMatrix.shape)
    # bTermcosinePart2 = np.dot(bTermsMatrix, startEndTermNearestNeighborMatrix)
    # bTermCosinePart2 = bTermcosinePart2.sum(axis=1,dtype=float)
    # print("bTermCosinePart2: ", bTermCosinePart2.shape)
    # bfinalCosine = bTermCosinePart1+bTermCosinePart2
    # print("bTermFinalCosine: ",bfinalCosine.shape)
    # # sortedBterms = np.sort(bfinalCosine, axis=None)
    # # arraySize = np.size(bfinalCosine,0)
    #
    # dictBterms = {}
    # for counter, item in enumerate(possibleBTerms):
    #     meshTermKey = item
    #     meshTermVal = bfinalCosine[counter]
    #     dictBterms[meshTermKey] = 0.5*meshTermVal
    #
    # sorted_BTerms = sorted(dictBterms.items(), key=operator.itemgetter(1), reverse=True)
    # for key, value in sorted_BTerms:
    #     meshTerm = key
    #     count = value
    #     with open(filesPath + os.sep+ year+ os.sep  + "final-sortedBTermsByCosine.txt", 'a') as the_file:
    #         prepareString = meshTerm + "\t" + str(count)
    #         the_file.write(prepareString)
    #         the_file.write("\n")

if __name__ == "__main__":
    main()
import os
import pathlib
import sys
import numpy as np
import scipy
import operator
import settings
from class_globaltransformationMatrix import class_globaltransformationMatrix
from class_NearestNeighbour import class_NearestNeighbour

def main():

    # print("there")
    # sys.exit(0)

    settings.init()
    classGlobaltransformationMatrixInit = class_globaltransformationMatrix(settings.dict['yearBase'],
                                                                           settings.dict['yearTarget'],
                                                                           settings.dict['totalFrequentPairs'])

    meshTermsInCurrentYear = []
    nearestNeighbour = class_NearestNeighbour(settings.dict['inputTerm1'], settings.dict['year'], meshTermsInCurrentYear)
    learnTransformationMatrix_M = classGlobaltransformationMatrixInit.globalTransformationMatrix()
    classGlobaltransformationMatrixInit.generateTransformationMatrix()
    print("learnGlobalTransformationMatrix_M: ",learnTransformationMatrix_M.shape)
    # sys.exit(0)

    startInputTermNearestNeighbor = []
    endInputTermNearestNeighbor = []

    for counter, item in enumerate(open(settings.dict['FILES_PATH'] + os.sep + settings.dict['year'] + os.sep +
                                                "neighbours-"+settings.dict['inputTerm1'] + "-sortedByCosine.txt")):
        itemSplit_input = item.split("\t")
        termName_input = itemSplit_input[0].lower()
        startInputTermNearestNeighbor.append(termName_input)
        if (counter > settings.dict['neighboursThreshold']):
            break;

    for counter, item in enumerate(open(settings.dict['FILES_PATH'] + os.sep + settings.dict['year'] + os.sep +
                                                "neighbours-"+settings.dict['inputTerm2'] + "-sortedByCosine.txt")):
        itemSplit_end = item.split("\t")
        termName_end = itemSplit_end[0].lower()
        endInputTermNearestNeighbor.append(termName_end)
        if (counter > settings.dict['neighboursThreshold']):
            break;

    print("startInputTermNearestNeighbor: ", len(startInputTermNearestNeighbor))
    print("endInputTermNearestNeighbor: ", len(endInputTermNearestNeighbor))

    # print("path: ", settings.dict['FILES_PATH'] + os.sep + str(int(settings.dict['year']) - 1))
    previousYearBTerms = []
    for counter, item in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "candidateBTerms" + os.sep + settings.dict['inputTerm1'] + "-" + settings.dict['inputTerm2'] +"-candidateBTerms")):
        # itemSplit_allMeSH = item.split("\t")
        previousYearBTerms.append(item.strip().lower())
        # if (len(itemSplit_allMeSH) > 0):
        #     meshTermName = itemSplit_allMeSH[0].lower().strip();
        #     previousYearBTerms.append(meshTermName.strip().lower())

    print("previousYearBTerms: ", len(previousYearBTerms))

    bTermCosineMatrix = np.empty([len(previousYearBTerms), 300])

    embeddingsNew = nearestNeighbour.loadEmbeddings(settings.dict['yearBase']);
    for counter, item in enumerate(previousYearBTerms):
        # print("B term:",item)
        if(item in startInputTermNearestNeighbor or item in endInputTermNearestNeighbor):
            continue
        else:
            embedding1 = nearestNeighbour.getEmbeddings(item, embeddingsNew)
            bTermCosineMatrix[counter] = np.dot(embedding1, learnTransformationMatrix_M)
            # path = pathlib.Path(
            #     settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + str(int(settings.dict['year']) - 1) + os.sep + item.lower()) #use base year
            # if (path.exists()):
            #
            #     embedding1 = nearestNeighbour.getEmbeddings(item, str(int(settings.dict['year']) - 1))
            #     # bTermCosineMatrix[counter] = embedding1
            #     bTermCosineMatrix[counter] = np.dot(embedding1, learnTransformationMatrix_M)

    print("BTermsMatrix: ", bTermCosineMatrix.shape)

    startTermNearestNeighborMatrix = np.empty([len(startInputTermNearestNeighbor), 300])
    embeddingsNew1 = nearestNeighbour.loadEmbeddings(settings.dict['yearTarget']);
    for counter2, item2 in enumerate(startInputTermNearestNeighbor):
        embedding2 = nearestNeighbour.getEmbeddings(item2, embeddingsNew1)
        # startTermNearestNeighborMatrix[counter] = np.dot(embedding2, learnTransformationMatrix_M)
        startTermNearestNeighborMatrix[counter2] = embedding2

        # path2 = pathlib.Path(
        #     settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + settings.dict['year'] + os.sep + item2.lower())
        #
        # if (path2.exists()):
        #     embedding3 = nearestNeighbour.getEmbeddings(item2, settings.dict['year'])
        #     startTermNearestNeighborMatrix[counter2] = embedding3

    print("startTermNearestNeighborMatrix: ", startTermNearestNeighborMatrix.shape)

    bTermCosinePart1 = 1 - scipy.spatial.distance.cdist(bTermCosineMatrix, startTermNearestNeighborMatrix, 'cosine')
    print("bTermCosinePart1: ", bTermCosinePart1.shape)

    bTermCosinePart1 = bTermCosinePart1.sum(axis=1,dtype=float)
    print("bTermCosinePart1_SUM: ",bTermCosinePart1.shape)

    startEndTermNearestNeighborMatrix = np.empty([len(endInputTermNearestNeighbor),300])
    embeddingsNew3 = nearestNeighbour.loadEmbeddings(settings.dict['yearTarget']);
    for counter3, item3 in enumerate(endInputTermNearestNeighbor):
        embedding3 = nearestNeighbour.getEmbeddings(item3, embeddingsNew3)
        # startTermNearestNeighborMatrix[counter] = np.dot(embedding2, learnTransformationMatrix_M)
        startEndTermNearestNeighborMatrix[counter3] = embedding3

        # path3 = pathlib.Path(settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + settings.dict['year'] + os.sep +item3.lower())
        #
        # if(path3.exists()):
        #     embedding4 = nearestNeighbour.getEmbeddings(item3, settings.dict['year'])
        #     startEndTermNearestNeighborMatrix[counter3] = embedding4

    print("startEndInputTermNearestNeighborMatrix: ", startEndTermNearestNeighborMatrix.shape)

    bTermCosinePart2 = 1 - scipy.spatial.distance.cdist(bTermCosineMatrix, startEndTermNearestNeighborMatrix, 'cosine')
    print("bTermCosinePart2: ",bTermCosinePart2.shape)

    bTermCosinePart2 = bTermCosinePart2.sum(axis=1,dtype=float)
    print("bTermCosinePart2_SUM: ",bTermCosinePart2.shape)

    bfinalCosine = bTermCosinePart1 + bTermCosinePart2
    print("bTermFinalCosine: ",bfinalCosine.shape)

    dictBterms = {}
    for counter, item in enumerate(previousYearBTerms):
        meshTermKey = item
        meshTermVal = bfinalCosine[counter]
        if (np.isnan(meshTermVal)):
            continue
        else:
            dictBterms[meshTermKey] = 0.5*meshTermVal #AB and BC so divide by 2

    sorted_BTerms = sorted(dictBterms.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sorted_BTerms:
        meshTerm = key
        count = value
        with open(settings.dict['FILES_PATH'] + os.sep+ settings.dict['year']+ os.sep  + "global-trans-final-sortedBTermsByCosine.txt", 'a') as the_file:
            prepareString = meshTerm + "\t" + str(count)
            the_file.write(prepareString)
            the_file.write("\n")
    print("writing of global transformation matrix completed")
if __name__ == "__main__":
    main()
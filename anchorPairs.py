import pathlib
import numpy as np
import scipy
from scipy.spatial import distance
import settings
from class_NearestNeighbour import class_NearestNeighbour
import os
import operator
import sys
import subprocess

settings.init()
meshTermsInCurrentYear = []

def getCosineThreshold(year, neighbourClass):
    loadEmbeddingOfParticularYear = neighbourClass.loadEmbeddings(year);
    embeddingInputTermOne = neighbourClass.getEmbeddings(settings.dict['inputTerm1'], loadEmbeddingOfParticularYear)
    embeddingInputTermTwo = neighbourClass.getEmbeddings(settings.dict['inputTerm2'], loadEmbeddingOfParticularYear)
    cosineThreshold = 1 - scipy.spatial.distance.cdist(embeddingInputTermOne.reshape(1, -1),
                                                       embeddingInputTermTwo.reshape(1, -1), 'cosine')
    print(cosineThreshold)
    return cosineThreshold[0][0]


def loadtopNTermsInputStartTerm(term, year):
    inputTermNearestNeighbor = []
    for counter, item in enumerate(
            open(settings.dict['FILES_PATH'] + os.sep + year + os.sep + "neighbours-" + term + "-sortedByCosine.txt")):
        itemSplit_input = item.split("\t")
        termName_input = itemSplit_input[0].lower()
        inputTermNearestNeighbor.append(termName_input)
        if counter > settings.dict['anchorPairsNumberThreshold']:
            break;
    return inputTermNearestNeighbor


def prepareDataMarix(topNTerms, year, neighbourClass):
    loadEmbedding = neighbourClass.loadEmbeddings(year)
    dataMatrix = np.empty([len(topNTerms), 300])
    for counter, item in enumerate(topNTerms):
        path = pathlib.Path(
            settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + year + os.sep + item.lower())

        if (path.exists()):
            embedding = neighbourClass.getEmbeddings(item, loadEmbedding)
            dataMatrix[counter] = embedding

    return dataMatrix


def deleteOldFiles(dirPath):
    for the_file in os.listdir(dirPath):
        file_path = os.path.join(dirPath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def generatePairCooccurence(sorted_Pairs, year, neighbourClass):
    loadEmbedding = neighbourClass.loadEmbeddings(year)
    dirPath = settings.dict['FILES_PATH'] + os.sep + year + os.sep + "pairCooccurence"
    deleteOldFiles(dirPath)
    for key1, value1 in sorted_Pairs:
        meshPairName1 = key1
        meshPairValue1 = value1
        # print(meshPairName1 + " " + str(meshPairValue1))

        splitMeshPairName1 = meshPairName1.split("$$$")
        meshPairName1_A = np.asarray(neighbourClass.getEmbeddings(splitMeshPairName1[0],loadEmbedding))
        meshPairName1_A = meshPairName1_A.astype('float64')
        meshPairName1_B = np.asarray(neighbourClass.getEmbeddings(splitMeshPairName1[1], loadEmbedding))
        meshPairName1_B  = meshPairName1_B.astype('float64')

        for key2, value2 in sorted_Pairs:
            if (key1 == key2):
                continue
            else:
                meshPairName2 = key2
                meshPairValue2 = value2
                # print(str(meshPairValue1)+"----"+str(meshPairValue2))

                splitMeshPairName2 = meshPairName2.split("$$$")
                meshPairName2_A = np.asarray(neighbourClass.getEmbeddings(splitMeshPairName2[0], loadEmbedding))
                meshPairName2_A = meshPairName2_A.astype('float64')
                meshPairName2_B = np.asarray(neighbourClass.getEmbeddings(splitMeshPairName2[1], loadEmbedding))
                meshPairName2_B = meshPairName2_B.astype('float64')
                meshPairVecDiff_A = np.subtract(meshPairName1_A , meshPairName2_A)
                meshPairVecDiff_B = np.subtract(meshPairName1_B , meshPairName2_B)
                relationSimilarity = 1 - scipy.spatial.distance.cdist(meshPairVecDiff_A.reshape(1, -1),
                                                                      meshPairVecDiff_B.reshape(1, -1), 'cosine')

                relationSimilarityValue = relationSimilarity[0][0]
                if (np.isnan(relationSimilarityValue)):
                    relationSimilarityValue = 0.0

                mesPairVal = 2.0 * (meshPairValue1 * meshPairValue2) / (meshPairValue1 + meshPairValue2)
                updatedValue2 = mesPairVal + relationSimilarityValue
                # updatedValue2 = meshPairValue1 + meshPairValue2 + relationSimilarityValue # May be use F1 Cosine here

                # updatedValue2 = relationSimilarityValue

                minPairValue = min(meshPairValue1, meshPairValue2)
                maxPairValue = max(meshPairValue1, meshPairValue2)
                threshold = minPairValue / maxPairValue
                # print(threshold)

                if threshold > settings.dict['anchorPairsOverlapThreshold']:
                    with open(settings.dict['FILES_PATH'] + os.sep + year + os.sep + "pairCooccurence" + os.sep + meshPairName1, 'a') as the_file:
                        prepareString = meshPairName2 + "\t" + str(updatedValue2)
                        the_file.write(prepareString)
                        the_file.write("\n")

    print("Pair cooccurence created")


def generateAnchorForYear(year, neighbourClass):

    cosineThreshold = getCosineThreshold(year, neighbourClass)
    topNTermsInputStartTerm = loadtopNTermsInputStartTerm(settings.dict['inputTerm1'], year)
    topNTermsInputEndTerm = loadtopNTermsInputStartTerm(settings.dict['inputTerm2'], year)
    # print(topNTermsInputStartTerm)
    # print(topNTermsInputEndTerm)

    dataMatrixOne = prepareDataMarix(topNTermsInputStartTerm, year, neighbourClass)
    dataMatrixTwo = prepareDataMarix(topNTermsInputEndTerm, year, neighbourClass)

    # print(dataMatrixOne.shape)
    # print(dataMatrixTwo.shape)

    cosineSim = 1 - scipy.spatial.distance.cdist(dataMatrixOne, dataMatrixTwo, 'cosine')
    # print(cosineSim)

    mainCounter = 0
    dictPairTerms = {}
    for counter, item in enumerate(topNTermsInputStartTerm):
        for counter1, item1 in enumerate(topNTermsInputEndTerm):
            value = cosineSim[counter][counter1]
            if (value > cosineThreshold):
                # print("Term pair: ",item+" "+item1+" "+str(value) + " " + str(mainCounter))
                mainCounter = mainCounter + 1
                termpair = item + "$$$" + item1
                dictPairTerms[termpair] = value

    sorted_Pairs = sorted(dictPairTerms.items(), key=operator.itemgetter(1), reverse=True)

    # for key1, value1 in sorted_Pairs:
    #     meshPairName1 = key1
    #     meshPairValue1 = value1
    #     print(meshPairName1+" "+str(meshPairValue1))

    generatePairCooccurence(sorted_Pairs, year, neighbourClass)
    print("Anchor pairs Python generated")
    subprocess.call(['java', '-jar', "/home/super-machine/NetBeansProjects/MeSHRanking/dist/MeSHRanking.jar", year])
    print("Anchor pairs Java generated")


def main():
     # neighbourClass
    year1 = settings.dict['year']
    neighbourClass1 = class_NearestNeighbour(settings.dict['inputTerm1'], year1, meshTermsInCurrentYear)
    generateAnchorForYear(year1, neighbourClass1)
    print("Anchors generated for year: ", year1)

    year2 = int(settings.dict['year']) -1
    year2 = str(year2)
    print("year2: ", str(year2))
    neighbourClass2 = class_NearestNeighbour(settings.dict['inputTerm1'], year2, meshTermsInCurrentYear)
    generateAnchorForYear(year2, neighbourClass2)
    print("Anchors generated for year: ", year2)


if __name__ == "__main__":
    main()

import os
import pathlib

import numpy as np
import settings
from class_NearestNeighbour import NearestNeighbour
import class_globaltransformationMatrix
from class_globaltransformationMatrix import globaltransformationMatrix

settings.init()
# print(settings.dict['ROOT_PATH'])
# meshTermsInCurrentYear = []
# nearestNeighbour = NearestNeighbour(settings.dict['inputTerm2'],  settings.dict['year'], meshTermsInCurrentYear)
# nearestNeighbour.loadMeshTermsInGivenYear();
# embeddingInputTerm1 = nearestNeighbour.getEmbeddings(settings.dict['inputTerm1'])
# print("meshTermsInCurrentYear: ", len(meshTermsInCurrentYear))
# nearestNeighbour.writeToFileTheNearestNeighbour(embeddingInputTerm1)

meshTermsInCurrentYear = []
nearestNeighbour = NearestNeighbour(settings.dict['inputTerm1'], settings.dict['year'], meshTermsInCurrentYear)

startInputTermNearestNeighbor = []
endInputTermNearestNeighbor = []

for counter, item in enumerate(open(settings.dict['FILES_PATH'] + os.sep + settings.dict['year'] + os.sep + settings.dict['inputTerm1'] + "-sortedByCosine.txt")):
    itemSplit_input = item.split("\t")
    termName_input = itemSplit_input[0].lower()
    startInputTermNearestNeighbor.append(termName_input)
    if (counter > 20):
        break;

for counter, item in enumerate(open(settings.dict['FILES_PATH']  + os.sep + settings.dict['year'] + os.sep + settings.dict['inputTerm1'] + "-sortedByCosine.txt")):
    itemSplit_end = item.split("\t")
    termName_end = itemSplit_end[0].lower()
    endInputTermNearestNeighbor.append(termName_end)
    if (counter > 20):
        break;

print("startInputTermNearestNeighbor: ", len(startInputTermNearestNeighbor))
print("endInputTermNearestNeighbor: ", len(endInputTermNearestNeighbor))

possibleBTerms = []
for counter, item in enumerate(open(settings.dict['FILES_PATH']+os.sep+str(int(settings.dict['year'])-1)+os.sep+"final-sortedBTermsByCosine.txt")):
    split_item = item.split("\t")
    possibleBTerms.append(split_item[0].lower())

print("Possible B Terms: ", len(possibleBTerms))

globaltransformationMatrixObj = globaltransformationMatrix(settings.dict['yearBase'], settings.dict['yearTarget'], settings.dict['totalFrequentPairs'])
globaltransformationMatrixObj.generateTransformationMatrix()
# bTermsMatrix = np.empty([len(possibleBTerms), 300])
# for counter1, item1 in enumerate(possibleBTerms):
#     # print("B term:",item)
#     path = pathlib.Path(settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + str(int(settings.dict['year'])-1) + os.sep + item1.lower())
#
#     if path.exists():
#         embedding1 = nearestNeighbour.getEmbeddings(item1)
#         # print(embedding1)
#         # learnTransformationMatrix_M = globaltransformationMatrix.globalTransformationMatrix()
#         bTermsMatrix[counter1] = embedding1
#
# print("BTermsMatrix: ", bTermsMatrix.shape)

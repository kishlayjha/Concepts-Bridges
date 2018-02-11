import numpy as np
import numpy.linalg as la
import scipy.stats
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse.linalg as sla

import settings

# ROOT_PATH = "/home/super-machine/Documents/mydrive/myResearch/output"
# FILES_PATH = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"

class class_globaltransformationMatrix:

    rootPath = "/home/super-machine/Documents/mydrive/myResearch/output"
    filesPath = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"
    meshDictionary = {}
    mesh_index = {}
    # embeddings = np.array((27802,300))

    def __init__(self, yearBase, yearTarget, totalFrequentPairs):
        settings.init();
        self.yearBase = yearBase
        self.yearTarget = yearTarget
        self.totalFrequentPairs = totalFrequentPairs
        self.loadMeshTermIntoDic()
        self.loadIndexOfMesh()
        # self.loadEmbeddings(self.year)

    def loadMeshTermIntoDic(self):
        for counter, line in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "mesh2017TreeMapping")):
            lineSplit = line.split("\t")
            if (len(lineSplit) > 0):
                meshTermName = lineSplit[0].lower().strip();
                meshCodes = []
                for counter1, line1 in enumerate(lineSplit):
                    if ((counter1 + 1) < len(lineSplit)):
                        meshCodes.append(lineSplit[counter1 + 1])
            self.meshDictionary[meshTermName] = meshCodes

    def cos_sim(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def loadIndexOfMesh(self):
        for counter, line in enumerate(open(self.filesPath + os.sep + "vocab.txt")):
            splitLine = line.split("\t")
            termIndex = splitLine[0].strip().lower()
            termName = splitLine[1].strip().lower()
            self.mesh_index[termName] = termIndex
        # print("mesh_index: ", len(self.mesh_index))
        return self.mesh_index

    def loadEmbeddings(self, year):
        embeddings = np.load(self.filesPath + os.sep + "embeddings" + os.sep + year + ".npy")
        return embeddings

    def getEmbeddings(self, termName, embeddingsNew):
        index = self.mesh_index.get(termName)
        wordVectorsArray = np.asarray(embeddingsNew[int(index)], dtype=float)
        return wordVectorsArray

    # def getEmbeddings(self, inputTerm, year):
    #     embeddingPath = settings.dict['ROOT_PATH'] + os.sep + "invertedIndex" + os.sep + year + os.sep + inputTerm
    #     wordVectorsArray = np.zeros(300, dtype=float)
    #     with open(embeddingPath, 'r') as myEmbeddingfile:
    #         wordVectorsString = myEmbeddingfile.read()
    #         wordVectors = wordVectorsString.split("\t")
    #         for index, vector in enumerate(wordVectors):
    #             wordVectorsArray[index] = vector
    #     return wordVectorsArray

    def least_square_optimization(self, baseMatrix_W, targetMatrix_W1, l):
        n, m = baseMatrix_W.shape
        I = np.identity(m)
        M = np.dot(np.dot(la.inv(np.dot(baseMatrix_W.transpose(), baseMatrix_W) + l * I), baseMatrix_W.transpose()), targetMatrix_W1)
        return M


    def prepareDataMarixOfCommonFrequentTerms(self, stableTermsInParticlarTimePeriod_X_Y, year):
        embeddingsNew = self.loadEmbeddings(year);
        data = "";
        for counter, term in enumerate(stableTermsInParticlarTimePeriod_X_Y):
            vector =  self.getEmbeddings(term, embeddingsNew)
            if counter == 0:
                deliminator = ""
            else:
                deliminator = ";"
            data = data + deliminator + str(vector).replace("[", "").replace("]", "")
        # print(data)
        datamatrix = np.matrix(data)
        return datamatrix

    # def prepareDataMarixOfCommonFrequentTerms(self, stableTermsInParticlarTimePeriod_X_Y, year):
    #     self.loadEmbeddings(year)
    #     data = "";
    #     for counter, term in enumerate(stableTermsInParticlarTimePeriod_X_Y):
    #         vector = self.getEmbeddings(term, year)
    #         if counter == 0:
    #             deliminator = ""
    #         else:
    #             deliminator = ";"
    #         data = data + deliminator + str(vector).replace("[", "").replace("]", "")
    #     # print(data)
    #     datamatrix = np.matrix(data)
    #     return datamatrix

    def prepareDataMarixOfLocalAnchorPairs(self, localAnchorPairs, year):
        embeddingsNew = self.loadEmbeddings(year);
        data = "";
        for counter, term in enumerate(localAnchorPairs):
            splitLocalAnchorPairs = term.split("$$$")
            anchorPairOne = splitLocalAnchorPairs[0]
            anchorPairTwo = splitLocalAnchorPairs[1]

            vectoranchorPairOne = np.asarray(self.getEmbeddings(anchorPairOne, embeddingsNew))
            vectoranchorPairOne = vectoranchorPairOne.astype('float64')
            vectoranchorPairTwo = np.asarray(self.getEmbeddings(anchorPairTwo, embeddingsNew))
            vectoranchorPairOne = vectoranchorPairOne.astype('float64')
            vector = np.subtract(vectoranchorPairOne, vectoranchorPairTwo)
            if counter == 0:
                deliminator = ""
            else:
                deliminator = ";"
            data = data + deliminator + str(vector).replace("[", "").replace("]", "")
        # print(data)
        datamatrix = np.matrix(data)
        return datamatrix

    def loadSemanticallyStableTerm(self, year):
        stableTermsInTimePeriod = []
        for counter, item in enumerate(
                open(settings.dict['FILES_PATH'] + os.sep + "semanticallyStableTerms" + os.sep + "2016")):
            if (counter > 0):
                termSplit = item.split("\t")
                stableTermsInTimePeriod.append(termSplit[0])
                if (counter > 600):
                    break
        return stableTermsInTimePeriod


    def loadLocalAnchorPairs(self, year):
        localanchorPairs= []
        for counter, item in enumerate(
                open(settings.dict['FILES_PATH'] + os.sep +year + os.sep + "anchorPairs-PageRank.txt")):
            if (counter > 0):
                termSplit = item.split("\t")
                localanchorPairs.append(termSplit[0].strip())
                if (counter > 600): # Anchor pair set is usually small so this is fine
                    break
        return localanchorPairs


    def localTransformationMatrix(self, dataMatrix_Base, dataMarix_Target):
        learnLocalTransformationMatrix = self.least_square_optimization(dataMatrix_Base, dataMarix_Target, 0.02)
        return learnLocalTransformationMatrix

    def globalTransformationMatrix(self):

        stableTermsInBaseTimePeriod_X = self.loadSemanticallyStableTerm(self.yearBase)
        stableTermsInTargetTimePeriod_Y = self.loadSemanticallyStableTerm(self.yearTarget)

        finalStableTerms = []
        for counter, index in enumerate(stableTermsInBaseTimePeriod_X):
            if (index in stableTermsInTargetTimePeriod_Y):
                finalStableTerms.append(index)
                if (counter > self.totalFrequentPairs):
                    break

        # print(finalStableTerms)
        matrixDataBaseTimePeriod_X = self.prepareDataMarixOfCommonFrequentTerms(finalStableTerms, self.yearBase)
        # print("Base Shape: ", str(matrixDataBaseTimePeriod_X.shape))

        matrixDataTargetTimePeriod_Y = self.prepareDataMarixOfCommonFrequentTerms(finalStableTerms, self.yearTarget)
        # print("Target Shape: ", str(matrixDataTargetTimePeriod_Y.shape))

        # learnTransformationMatrix_M = np.matrix(least_square_optimization(matrixDataBaseTimePeriod_X,matrixDataTargetTimePeriod_Y,0.02))
        learnTransformationMatrix_M = self.least_square_optimization(matrixDataBaseTimePeriod_X,
                                                                     matrixDataTargetTimePeriod_Y, 0.02)
        return learnTransformationMatrix_M

    def generateTransformationMatrix(self):
        # self.loadMeshTermIntoDic();

        # print(stableTermsInBaseTimePeriod_X)
        # print(stableTermsInTargetTimePeriod_Y)
        learnTransformationMatrix_M = self.globalTransformationMatrix()
        print("Learned Transformation Matrix", learnTransformationMatrix_M.shape)
        # return learnTransformationMatrix_M
        input1 = "blood viscosity"
        input2 = "fish oils"
        # term1 = self.getEmbeddings(input1, "1985")
        # term2 = self.getEmbeddings(input2, "1985")
        embeddingsNew1 = self.loadEmbeddings(self.yearBase);
        embeddingsNew2 = self.loadEmbeddings(self.yearTarget);
        term1 = self.getEmbeddings(input1, embeddingsNew1)
        term2 = self.getEmbeddings(input2, embeddingsNew2)
        finalCosineScore = self.cos_sim(np.dot(term1, learnTransformationMatrix_M), term2)
        # finalCosineScore = self.cos_sim(term1, term2)
        print("finalCosineScore: ", str(finalCosineScore))

        # if __name__ == "__main__":
        #     main()

import numpy as np
import os
import scipy
from scipy.spatial import distance
from pathlib import Path

class class_NearestNeighbour:

    rootPath = "/home/super-machine/Documents/mydrive/myResearch/output"
    filesPath = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"
    mesh_index = {}
    # embeddings = np.array((27802,300))


    def __init__(self, inputTerm, year, meshTermsInCurrentYear):
        self.inputTerm = inputTerm
        self.year = year
        self.meshTermsInCurrentYear = meshTermsInCurrentYear
        self.loadIndexOfMesh()
        # self.loadEmbeddings(self.year)

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


    # def getEmbeddings(self, termName, year):
    #     embeddingPath = self.rootPath + os.sep + "invertedIndex" + os.sep + year + os.sep + termName
    #     wordVectorsArray = np.zeros(300, dtype=float)
    #     with open(embeddingPath, 'r') as myEmbeddingfile:
    #         wordVectorsString = myEmbeddingfile.read()
    #         wordVectors = wordVectorsString.split("\t")
    #         for index, vector in enumerate(wordVectors):
    #             wordVectorsArray[index] = vector
    #     return wordVectorsArray

    def loadMeshTermsInGivenYear(self):
        for filename in os.listdir(self.rootPath + os.sep + "invertedIndex" + os.sep + self.year):
            self.meshTermsInCurrentYear.append(filename)

    def checkSemTypeWithInput(self, term):
        semTypesOfTerm = []
        my_file = Path(self.rootPath + os.sep + "invertedSemanticMeshIndex"+ os.sep + term)
        if my_file.is_file():
            for counter, line in enumerate(open(self.rootPath + os.sep + "invertedSemanticMeshIndex"+ os.sep + term)):
                if line == "\n":
                    continue
                else:
                    semTypesOfTerm.append(line)
        return semTypesOfTerm


    def writeToFileTheNearestNeighbour(self, embeddingOfInputTerm1, year):
        unsortedDic = {}
        getSemTypeOfInputTerm = self.checkSemTypeWithInput(self.inputTerm)
        # print("getSemTypeOfInputTerm: ",getSemTypeOfInputTerm)
        embeddingsNew = self.loadEmbeddings(year);
        # print(embeddingsNew.shape)
        for counter, termName in enumerate(self.meshTermsInCurrentYear):
            if (self.inputTerm.lower() != termName):
                getSemTypeOfTerm = self.checkSemTypeWithInput(termName);
                checkSemType = False
                for count, semType in enumerate(getSemTypeOfTerm):
                    # print("SemType: ",semType)
                    if semType in getSemTypeOfInputTerm:
                        checkSemType = True
                        break

                if(checkSemType):
                    # cosineScore = self.cos_sim(embeddingOfInputTerm1, embeddingInputTerm2)
                    embeddingInputTerm2 = self.getEmbeddings(termName, embeddingsNew)
                    # print("embeddingInputTerm2: ",embeddingInputTerm2.shape)
                    cosineScore = scipy.spatial.distance.cosine(embeddingOfInputTerm1.reshape(1,-1), embeddingInputTerm2.reshape(1,-1))
                    # print("CosineScore",cosineScore," ",termName, " "+str(counter))
                    if (np.isnan(cosineScore)):
                        continue
                    else:
                        unsortedDic[termName] = cosineScore

                # print("getSemTypeOfIntermediateTerm: ", getSemTypeOfTerm)

        for key, value in sorted(unsortedDic.items(), key=lambda x: (x[1], x[0]), reverse=False):
            with open(self.filesPath+os.sep+self.year+os.sep+"neighbours-"+self.inputTerm +"-sortedByCosine.txt", 'a') as the_file:
                prepareString = key + "\t" + str(value)
                the_file.write(prepareString)
                the_file.write("\n")
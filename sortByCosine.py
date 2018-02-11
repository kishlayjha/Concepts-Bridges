import os
import numpy as np


inputTerm1 = "raynaud disease"
year = "1985"
ROOT_PATH = "/home/super-machine/Documents/mydrive/myResearch/output"
meshTermsInCurrentYear = []

def cos_sim(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)


def getEmbeddings(inputTerm, year):
    embeddingPath = ROOT_PATH+os.sep+"invertedIndex"+os.sep+year+os.sep+inputTerm
    wordVectorsArray = np.zeros(300,dtype=float)
    with open(embeddingPath, mode='r') as myEmbeddingfile:
        wordVectorsString = myEmbeddingfile.read()
        wordVectors = wordVectorsString.split("\t")
        for index, vector in enumerate(wordVectors):
            wordVectorsArray[index] = vector
    return wordVectorsArray


def loadMeshTermsInGivenYear(year):
        for filename in os.listdir(ROOT_PATH+os.sep+"invertedIndex"+os.sep+year):
            # print(filename)
            meshTermsInCurrentYear.append(filename)
        print("meshTermsInCurrentYear: ",len(meshTermsInCurrentYear))

def writeToFile(meshTerm, cosineScore):
    with open("sortedByCosine.txt",'a') as the_file:
        the_file.write(meshTerm + "\t" + cosineScore)
        the_file.write("\n")

def main():
    loadMeshTermsInGivenYear(year);
    embeddingInputTerm1 = getEmbeddings(inputTerm1,year)
    # embeddingInputTerm2 = getEmbeddings("pichinde virus", year)
    # print("embeddingInputTerm1: ",str(embeddingInputTerm1))
    # print("embeddingInputTerm2: ", str(embeddingInputTerm2))
    # cosineScore = cos_sim(embeddingInputTerm1, embeddingInputTerm2)
    # print("Cosine val:", str(cosineScore))

    unsortedDic = {}
    for counter, index in enumerate(meshTermsInCurrentYear):
        # print(index)
        if(inputTerm1.lower() !=  index):
            print("index: ",index)
            embeddingInputTerm2 = getEmbeddings(index,year)
            cosineScore = cos_sim(embeddingInputTerm1,embeddingInputTerm2)
            # print(cosineScore)
            if(np.isnan(cosineScore)):
                continue
                # print(index)
            else:
                # print("index: ",str(index))
                print("Cosine val:",str(cosineScore))
                unsortedDic[index] = cosineScore

    for key, value in sorted(unsortedDic.items(), key=lambda x: (x[1], x[0]), reverse=True):
        # print(key,value)
        # writeToFile(key,value)
        with open("sortedByCosine.txt", 'a') as the_file:
            prepareString = key + "\t" + str(value)
            the_file.write(prepareString)
            the_file.write("\n")


if __name__ == "__main__":
    main()
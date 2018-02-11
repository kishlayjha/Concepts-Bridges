import settings
from class_NearestNeighbour import class_NearestNeighbour
import numpy as np
import os

settings.init()
mesh_index = {}


def loadEmbeddings(year):
    embeddings = np.load(settings.dict['FILES_PATH'] + os.sep + "embeddings" + os.sep + year + ".npy")
    return embeddings


def loadIndexOfMesh():
    for counter, line in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "vocab.txt")):
        splitLine = line.split("\t")
        termIndex = splitLine[0].strip().lower()
        termName = splitLine[1].strip()
        mesh_index[termName] = termIndex


def main_neighbour(embeddings, input, year):
    print("Input: ", input, " ", year)
    meshTermsInCurrentYear = []
    nearestNeighbour = class_NearestNeighbour(input, year,
                                              meshTermsInCurrentYear)
    nearestNeighbour.loadMeshTermsInGivenYear();
    embeddingInputTerm1 = np.asarray(embeddings[int(mesh_index.get(input))]);
    # print("embeddingInputTerm1",embeddingInputTerm1.shape)
    nearestNeighbour.writeToFileTheNearestNeighbour(embeddingInputTerm1, year)
    print("writing completed")


def main():

    loadIndexOfMesh()
    year1 = settings.dict['year']
    embeddings = loadEmbeddings(year1)
    main_neighbour(embeddings, settings.dict['inputTerm1'], year1)
    main_neighbour(embeddings, settings.dict['inputTerm2'], year1)
    print("Neighbours created for year: ", year1)
    year2 = int(settings.dict['year']) -1
    year2 = str(year2)
    embeddings_year2 = loadEmbeddings(year2)
    main_neighbour(embeddings_year2, settings.dict['inputTerm1'], year2)
    main_neighbour(embeddings_year2, settings.dict['inputTerm2'], year2)
    print("Neighbours created for year: ", year2)

# meshTermsInCurrentYear = []
# nearestNeighbour = class_NearestNeighbour(settings.dict['inputTerm1'], settings.dict['year'], meshTermsInCurrentYear)
# nearestNeighbour.loadMeshTermsInGivenYear();
# print("meshTermsInCurrentYear: ", len(meshTermsInCurrentYear))
# embeddingInputTerm1 = nearestNeighbour.getEmbeddings(settings.dict['inputTerm1'], settings.dict['year'])
# nearestNeighbour.writeToFileTheNearestNeighbour(embeddingInputTerm1, settings.dict['year'])
# print("writing completed")

if __name__ == "__main__":
    main()

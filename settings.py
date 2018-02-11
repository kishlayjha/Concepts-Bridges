# import os

# mesh_index = {}
def init():
    # global mesh_index
    # mesh_index= loadIndexOfMesh()
    global dict
    dict = {}
    dict['ROOT_PATH']= "/home/super-machine/Documents/mydrive/myResearch/output"
    dict['FILES_PATH'] = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files"
    # meshDictionary = {}
    dict['baseline'] = "bicob"
    dict['totalFrequentPairs'] = 500
    dict['neighboursThreshold'] = 45
    dict['anchorPairsNumberThreshold'] = 50
    dict['anchorPairsOverlapThreshold'] = 0.50

    # dict['inputTerm1'] = "fish oils"
    # dict['inputTerm2'] = "raynaud disease"
    # dict['yearBase'] = "1984"
    # dict['yearTarget'] = "1985"
    # dict['year'] = "1985"
    # dict['candidateBTermFileName'] = "fish oils-raynaud disease-candidateBTerms"

    # dict['inputTerm1'] = "migraine disorders"
    # dict['inputTerm2'] = "magnesium"
    # dict['yearBase'] = "1987"
    # dict['yearTarget'] = "1988"
    # dict['year'] = "1988"
    # dict['candidateBTermFileName'] = "migraine disorders-magnesium-candidateBTerms"

    # dict['inputTerm1'] = "indomethacin"
    # dict['inputTerm2'] = "alzheimer disease"
    # dict['yearBase'] = "1994"
    # dict['yearTarget'] = "1993"
    # dict['year'] = "1994"
    # dict['candidateBTermFileName'] = "indomethacin-alzheimer disease-candidateBTerms"

    # dict['inputTerm1'] = "schizophrenia"
    # dict['inputTerm2'] = "phospholipases"
    # dict['yearBase'] = "1997"
    # dict['yearTarget'] = "1996"
    # dict['year'] = "1997"
    # dict['candidateBTermFileName'] = "schizophrenia-phospholipases-candidateBTerms"

    dict['inputTerm1'] = "insulin-like growth factor i"
    dict['inputTerm2'] = "arginine"
    dict['yearBase'] = "1988"
    dict['yearTarget'] = "1989"
    dict['year'] = "1989"
    dict['candidateBTermFileName'] = "insulin-like growth factor i-arginine-candidateBTerms"

# def loadIndexOfMesh():
#     for counter, line in enumerate(open("/home/super-machine/PycharmProjects/EmbeddingTransformation/files" + os.sep + "vocab.txt")):
#         splitLine = line.split("\t")
#         termIndex = splitLine[0].strip().lower()
#         termName = splitLine[1].strip()
#         mesh_index[termName] = termIndex
#     return mesh_index
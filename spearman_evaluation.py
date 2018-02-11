import numpy as np
import scipy.stats
import os
import settings
import sys

dict_ground_truth = {}
dict_baseline = {}
topKSpearman = 100
# groundTruthfileName = "groundtruth-fish oils-raynaud disease.txt"
groundTruthfileName = "groundtruth-fishoils-raynauddisease-kishlay3.txt"
# baseLineName = "BICOB"
# baselineFileName = "ranked_output_fishOils_raynaudsDisease.txt"
# baseLineName = "KDD"
# baselineFileName = "global6-and-Local-trans-final-sortedBTermsByCosine.txt"
baseLineName = "icdm"
baselineFileName = "fish oils-raynaud disease-constraint.txt"

def loadGroundTruthInDictionary():
    for counter, index in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "groundtruths" + os.sep + groundTruthfileName)):
        splitIndex = index.split("\t")
        term = splitIndex[0].lower()
        dict_ground_truth[term] = counter
    return dict_ground_truth

def loadBaseLineResultInDictionary(baseLineName):

    if baseLineName == "BICOB":
        for counter, index in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "baselines" + os.sep + "bicob" + os.sep + "glodenTestCasesresults" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("---")
            splitIndex1 = splitIndex[1]
            splitIndex2 = splitIndex1.split("[")
            term = splitIndex2[0].strip().lower()
            # print(term)
            dict_baseline[term] = counter

    if baseLineName == "KDD":
        for counter, index in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "baselines" + os.sep + "KDD" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            # print(term)
            dict_baseline[term] = counter

    if baseLineName == "icdm":
        for counter, index in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "baselines" + os.sep + "icdm" + os.sep + "goldenTestCasesresults" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            # print(term)
            dict_baseline[term] = counter

    return dict_baseline

def main():

    settings.init()
    dict_ground_truth =  loadGroundTruthInDictionary()
    print("dict_ground_truth: ",len(dict_ground_truth))
    dict_baseline = loadBaseLineResultInDictionary(baseLineName)
    print("dict_baseline: ",len(dict_baseline))
    # sys.exit(0)

    count = 1;
    groundTruthValues_List1 = []
    baselineValues_List2 = []
    for counter, index in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "groundtruths" + os.sep + groundTruthfileName)):
        splitIndex = index.split("\t")
        meshTerm = splitIndex[0].lower()
        # print(meshTerm)
        # print(dict_baseline.keys())
        if meshTerm in dict_baseline.keys():
            # print ("coming")
            baselineKeyScore = dict_baseline.get(meshTerm)
            groundTruthKeyScore = dict_ground_truth.get(meshTerm)
            print(str(count)+ " "+str(groundTruthKeyScore) + " "+ str(baselineKeyScore)+" "+meshTerm)
            groundTruthValues_List1.append(groundTruthKeyScore)
            baselineValues_List2.append(baselineKeyScore)
            count = count+1
            if(count > topKSpearman):
                break

    # print("count: ",count)
    print("groundTruthValues_List1: ",len(groundTruthValues_List1))
    print("baselineValues_List2: ",len(baselineValues_List2))

    # sys.exit(0)

    spearmanCoefficient = scipy.stats.mstats.spearmanr(np.asarray(groundTruthValues_List1),np.asarray(baselineValues_List2))
    print(spearmanCoefficient)

if __name__ == "__main__":
    main()
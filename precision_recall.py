import settings
import os

genericTerms = []

def loadGroundTruthFile(groundTruthfileName):
    groundTruthTerms = []
    for counter, index in enumerate(
            open(settings.dict['FILES_PATH'] + os.sep + "groundtruths" + os.sep + groundTruthfileName)):
        splitIndex = index.split("\t")
        term = splitIndex[0].lower()
        if term in genericTerms:
            continue
        else:
            groundTruthTerms.append(term)
    return groundTruthTerms


def calculatePrecisionAtTopK(baseTerms, groundTruth_Terms, topK):
    linkingTermCount = 0
    topKCounter = 0;
    for term in baseTerms:
        if term in groundTruth_Terms:
            linkingTermCount = linkingTermCount + 1
        topKCounter = topKCounter + 1
        if topKCounter > topK:
            break

    # print("Precision True positives: ", str(linkingTermCount))
    precisionAtTopK = linkingTermCount / topKCounter
    # print("precision@Topk: ", str(precisionAtTopK))
    return precisionAtTopK


def calculateRecallAtTopK(baseTerms, groundTruth_Terms, topK):
    linkingTermCount = 0
    topKCounter = 0;

    for term in baseTerms:
        if term in groundTruth_Terms:
            linkingTermCount = linkingTermCount + 1
        topKCounter = topKCounter + 1
        if topKCounter > topK:
            break

    # print("Recall True positives: ", str(linkingTermCount))
    recallAtTopK = linkingTermCount / len(groundTruth_Terms)
    # print("recall@TopK: ", str(recallAtTopK))
    return recallAtTopK


def calculateOverallPrecision(baseTerms, groundTruth_Terms):
    truePositive = 0
    for term in baseTerms:
        if term in groundTruth_Terms:
            truePositive = truePositive + 1
    print("True positive: ", truePositive)
    precision = truePositive / len(baseTerms)
    print("Overall Precision: ", precision)
    return precision


def calculateOverallRecall(baseTerms, groundTruth_Terms):
    truePositive = 0
    for term in groundTruth_Terms:
        if term in baseTerms:
            truePositive = truePositive + 1
    print("True positive: ", truePositive)
    recall = truePositive / len(groundTruth_Terms)
    print("Overall Recall: ", recall)
    return recall

def calculateMeanAveragePrecisionAtTopKGroundTruth(baseTerms, groundTruth_Terms, topK):
    # print("Base Terms Size:", len(baseTerms))
    # print("Ground Truth Terms Size:", len(groundTruth_Terms))
    baseTermCounter = 0
    groundTruthTermCounter = 0
    meanAverageprecision = 0
    truepositive = 0
    for term in groundTruth_Terms:

        if (groundTruthTermCounter > topK):
            break

        groundTruthTermCounter = groundTruthTermCounter + 1
        if (term in baseTerms):
            truepositive = truepositive + 1
            baseTermCounter = baseTermCounter + 1
            tempPrecision = truepositive / groundTruthTermCounter
            meanAverageprecision = meanAverageprecision + tempPrecision

    if baseTermCounter == 0:
        return 0
    else:
        finalMeanAvgPrecision = meanAverageprecision / baseTermCounter
    return finalMeanAvgPrecision


def calculateMeanAveragePrecisionAtTopKTargetTerm(baseTerms, groundTruth_Terms, topK):
    # print("Base Terms Size:", len(baseTerms))
    # print("Ground Truth Terms Size:", len(groundTruth_Terms))

    baseTermCounter = 0
    groundTruthTermCounter = 0
    meanAverageprecision = 0
    truepositive = 0
    # finalMeanAvgPrecision = 0
    for term in baseTerms:
        # print("Term: ",term)
        if (baseTermCounter > topK):
            break

        baseTermCounter = baseTermCounter + 1
        if (term in groundTruth_Terms):
            truepositive = truepositive + 1
            groundTruthTermCounter = groundTruthTermCounter + 1
            tempPrecision = truepositive / baseTermCounter
            meanAverageprecision = meanAverageprecision + tempPrecision
    # print(groundTruthTermCounter)
    finalMeanAvgPrecision = meanAverageprecision / groundTruthTermCounter
    return finalMeanAvgPrecision


def loadgenericTerms():
    genericCounter = 0
    for counter, index in enumerate(
            open(settings.dict['FILES_PATH'] + os.sep + "semanticallyStableTerms" + os.sep + "2016")):
        splitIndex = index.split("\t")
        term = splitIndex[0].lower()
        genericTerms.append(term)
        genericCounter = genericCounter + 1

        if genericCounter > 100:
            break
    return genericTerms


def main():
    settings.init()
    loadgenericTerms()

    for testCaseCounter, testCaseNames in enumerate(open(settings.dict['FILES_PATH'] + os.sep + "goldenTestCases.txt")):

        testCaseNames = testCaseNames.strip()
        splitTestCaseNames = testCaseNames.split("\t")
        inputTermName = splitTestCaseNames[1]+"-"+splitTestCaseNames[2]
        testCaseName = inputTermName
        groundTruthfileName = "ground-truth-"+testCaseName
        groundTruth_Terms = loadGroundTruthFile(groundTruthfileName)
        print("testCaseName ", testCaseName)
        print("*****************************")
        allbaselines = ['asr', 'chi-square', 'TF-IDF', 'mi', 'coh', 'bicob', 'cosine', 'icdm', 'KDD']
        # allbaselines = ['asr', 'TF-IDF', 'mi', 'coh', 'bicob', 'cosine', 'icdm', 'KDD']
        # allbaselines = ['coh','cosine', 'KDD']
        # allbaselines = ['bicob']

        for index in range(len(allbaselines)):
            baselineTerms = []
            algoName = allbaselines[index] + " & "
            # algoName = allbaselines[index] + "\t"
            topkPrecisionString = ""
            # for topKCounter in range(5, 30, 5):
            # for topKCounter in range(5, 55, 10):
            # for topKCounter in range(10, 60, 10):
            for topKCounter in range(9, 55, 10):
                baselinePathDir = settings.dict['FILES_PATH'] + os.sep + "baselines" + os.sep + allbaselines[index] + os.sep + testCaseName
                for fileName in os.listdir(baselinePathDir):
                    # print(fileName)
                    # filePath = baselinePathDir + os.sep + fileName
                    for counter, index1 in enumerate(open(str(baselinePathDir) + os.sep + str(fileName))):
                        index1 = index1.strip()
                        splitIndex = index1.split("\t")
                        term = splitIndex[0].strip().lower()
                        if term in genericTerms:
                            continue
                        else:
                            baselineTerms.append(term)
                    # print(algoName, baselineTerms)
                    precisionAtTopK = calculatePrecisionAtTopK(baselineTerms, groundTruth_Terms, topKCounter)
                    # print("precisionAtTopK",precisionAtTopK)
                    # recallAtTopK = calculateRecallAtTopK(baselineTerms, groundTruth_Terms, topKCounter)
                    # meanAveragePrecisionAtTopKTargetTerm = calculateMeanAveragePrecisionAtTopKTargetTerm(baselineTerms, groundTruth_Terms, topKCounter)

                topkPrecisionString = topkPrecisionString + str(round(precisionAtTopK, 3)) + " & "
                # topkPrecisionString = topkPrecisionString + str(round(meanAveragePrecisionAtTopKTargetTerm, 3)) + " & "
            # finalString = algoName + topkPrecisionString
            finalString = algoName + topkPrecisionString
            print(finalString)
        # sys.exit(-1)

        # fscoreATopK = (2.0 * float(precisionAtTopK) * float(recallAtTopK)) / (
        #     float(precisionAtTopK) + float(recallAtTopK))
        # print("fscoreATopK: ", fscoreATopK)
        # print("*************")

        # meanAveragePrecisionAtTopKTargetTerm = calculateMeanAveragePrecisionAtTopKTargetTerm(baseTerms, groundTruth_Terms)
        # print("*************")
        # meanAveragePrecisionAtTopKGroundTruth = calculateMeanAveragePrecisionAtTopKGroundTruth(baseTerms, groundTruth_Terms)


if __name__ == "__main__":
    main()

import settings
import os

topK = 25
topKMAP = 200
testCaseName = "fo-rd"

print("TopKPrecision: ", topK)
print("testCaseName ",testCaseName)
print("TopKMapPrecision: ", topKMAP)
# groundTruthfileName = "groundtruth-fish oils-raynaud disease.txt"
groundTruthfileName = "groundtruth-fishoils-raynauddisease-kishlay3.txt"
baseLineName = "BICOB"
baselineFileName = "ranked_output_fishOils_raynaudsDisease.txt"
# baseLineName = "KDD"
# baselineFileName = "global-and-local-final-sortedBTermsByCosine.txt"
# baseLineName = "icdm"
# baselineFileName = "fish oils-raynaud disease-constraint.txt"
# baseLineName = "asr"
# baselineFileName = "sorted-asr-raynaud diseasefish oils"
# baseLineName = "coh"
# baselineFileName = "sorted-coh-raynaud diseasefish oils"
# baseLineName = "TF-IDF"
# baselineFileName = "sorted-TF-IDF-raynaud diseasefish oils"
# baseLineName = "cosine"
# baselineFileName = "sorted-cosine-raynaud diseasefish oils"
# baseLineName = "chi-square"
# baselineFileName = "sorted-chi-square-raynaud diseasefish oils"

print("baseLineName: ", baseLineName)


def loadGroundTruthFile(groundTruthfileName):
    groundTruthTerms = []
    for counter, index in enumerate(
            open(settings.dict['FILES_PATH'] + os.sep + "groundtruths" + os.sep + groundTruthfileName)):
        splitIndex = index.split("\t")
        term = splitIndex[0].lower()
        groundTruthTerms.append(term)
    return groundTruthTerms


def loadBaselineTermsIntoList(baseLineName):
    baselineTerms = []
    if baseLineName == "BICOB":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "bicob" + os.sep + "glodenTestCasesresults" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("---")
            splitIndex1 = splitIndex[1]
            splitIndex2 = splitIndex1.split("[")
            term = splitIndex2[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "KDD":
        for counter, index in enumerate(
                open(settings.dict['FILES_PATH'] + os.sep + "baselines" + os.sep + "KDD" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "icdm":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "icdm" + os.sep + "goldenTestCasesresults" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "asr":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "asr" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "coh":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "coh" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "TF-IDF":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "TF-IDF" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "cosine":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "cosine" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    if baseLineName == "chi-square":
        for counter, index in enumerate(open(settings.dict[
                                                 'FILES_PATH'] + os.sep + "baselines" + os.sep + "chi-square" + os.sep + baselineFileName)):
            index = index.strip()
            splitIndex = index.split("\t")
            term = splitIndex[0].strip().lower()
            baselineTerms.append(term)

    return baselineTerms


def calculatePrecisionAtTopK(baseTerms, groundTruth_Terms):
    linkingTermCount = 0
    topKCounter = 0;
    for term in baseTerms:
        if term in groundTruth_Terms:
            linkingTermCount = linkingTermCount + 1
        topKCounter = topKCounter + 1
        if topKCounter > topK:
            break

    print("Precision True positives: ", str(linkingTermCount))
    precisionAtTopK = linkingTermCount / topKCounter
    print("precision@Topk: ", str(precisionAtTopK))
    return precisionAtTopK


def calculateRecallAtTopK(baseTerms, groundTruth_Terms):
    linkingTermCount = 0
    topKCounter = 0;

    for term in baseTerms:
        if term in groundTruth_Terms:
            linkingTermCount = linkingTermCount + 1
        topKCounter = topKCounter + 1
        if topKCounter > topK:
            break

    print("Recall True positives: ", str(linkingTermCount))
    recallAtTopK = linkingTermCount / len(groundTruth_Terms)
    print("recall@TopK: ", str(recallAtTopK))
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


def calculateMeanAveragePrecisionAtTopKTargetTerm(baseTerms, groundTruth_Terms):
    print("Base Terms Size:", len(baseTerms))
    print("Ground Truth Terms Size:", len(groundTruth_Terms))

    baseTermCounter = 0
    groundTruthTermCounter = 0
    meanAverageprecision = 0
    truepositive = 0
    for term in baseTerms:
        # print("Term: ",term)
        baseTermCounter = baseTermCounter + 1
        if (term in groundTruth_Terms):
            truepositive = truepositive + 1
            groundTruthTermCounter = groundTruthTermCounter + 1
            tempPrecision = truepositive / baseTermCounter
            meanAverageprecision = meanAverageprecision + tempPrecision
            finalMeanAvgPrecision = meanAverageprecision / groundTruthTermCounter
            # print("groundTruthTermCounter: ",str(groundTruthTermCounter))
            # print("finalMeanAvgPrecision@TargetTerm: ",str(finalMeanAvgPrecision))
        if (baseTermCounter > topKMAP):
            # print("baseTermCounter",baseTermCounter)
            break
    print("groundTruthTermCounter: ", str(groundTruthTermCounter))
    print("baseTermCounter", baseTermCounter)
    print("finalMeanAvgPrecision@TargetTerm: ", str(finalMeanAvgPrecision))


def calculateMeanAveragePrecisionAtTopKGroundTruth(baseTerms, groundTruth_Terms):
    print("Base Terms Size:", len(baseTerms))
    print("Ground Truth Terms Size:", len(groundTruth_Terms))

    baseTermCounter = 0
    groundTruthTermCounter = 0
    meanAverageprecision = 0
    truepositive = 0
    for term in groundTruth_Terms:
        # print("Term: ",term)
        groundTruthTermCounter = groundTruthTermCounter + 1
        if (term in baseTerms):
            truepositive = truepositive + 1
            baseTermCounter = baseTermCounter + 1
            tempPrecision = truepositive / groundTruthTermCounter
            meanAverageprecision = meanAverageprecision + tempPrecision
            finalMeanAvgPrecision = meanAverageprecision / baseTermCounter
            # print("groundTruthTermCounter: ",str(groundTruthTermCounter))
            # print("finalMeanAvgPrecision@TargetTerm: ",str(finalMeanAvgPrecision))
        if (groundTruthTermCounter > topKMAP):
            # print("baseTermCounter",baseTermCounter)
            break
    print("groundTruthTermCounter: ", str(groundTruthTermCounter))
    # print("baseTermCounter", baseTermCounter)
    print("finalMeanAvgPrecision@GroundTruthTerm: ", str(finalMeanAvgPrecision))


def main():
    settings.init()
    groundTruth_Terms = loadGroundTruthFile(groundTruthfileName)
    baseTerms = loadBaselineTermsIntoList(baseLineName)
    # print("*****")
    # print("baseTerms: ", len(baseTerms))
    # print("groundTruthTerms: ", len(groundTruth_Terms))

    allbaselines = ['asr', 'bicob', 'chi-square',  'coh',  'cosine',  'icdm', 'KDD', 'mi', 'TF-IDF']

    for index in range(len(allbaselines)):
        print('Current baseline :', allbaselines[index])
        pathdir = settings.dict['FILES_PATH'] + os.sep + "baselines"
        for i in range(5, 50, 5):
            print(i)


    precision = calculateOverallPrecision(baseTerms, groundTruth_Terms)
    recall = calculateOverallRecall(baseTerms, groundTruth_Terms)
    fscore = (2.0 * float(precision) * float(recall)) / (float(precision) + float(recall))
    print("F-score: ", fscore)
    print("*********************")
    precisionAtTopK = calculatePrecisionAtTopK(baseTerms, groundTruth_Terms)
    recallAtTopK = calculateRecallAtTopK(baseTerms, groundTruth_Terms)
    fscoreATopK = (2.0 * float(precisionAtTopK) * float(recallAtTopK)) / (float(precisionAtTopK) + float(recallAtTopK))
    print("fscoreATopK: ", fscoreATopK)
    print("*************")

    # meanAveragePrecisionAtTopKTargetTerm = calculateMeanAveragePrecisionAtTopKTargetTerm(baseTerms, groundTruth_Terms)
    # print("*************")
    # meanAveragePrecisionAtTopKGroundTruth = calculateMeanAveragePrecisionAtTopKGroundTruth(baseTerms, groundTruth_Terms)


if __name__ == "__main__":
    main()

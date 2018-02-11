import  os
import operator

ROOT_PATH = "/home/super-machine/Documents/mydrive/myResearch/output";
year1 = "1985"
year2 = "1986"

frequentMesh = {}

def findMostfrequentWordsInYear(year):
    for counter, item in enumerate(open(ROOT_PATH+os.sep+"PostCooccur"+os.sep+year)):
        meshPair = item.split("\t");
        meshTerm1 = meshPair[0]
        meshTerm2 = meshPair[1]
        # print(meshTerm1+"\t"+meshTerm2)
        if(meshTerm1.lower() == meshTerm2.lower()):
            # print("coming")
            cooccurVal = meshPair[2]
            cooccurValSplit = cooccurVal.split(".")
            frequentMesh[meshTerm1] = cooccurValSplit[0]
    return frequentMesh;


def writeToFile(frequentMeshTermsYear, year):
    # sorted_frequentMeshTermsYear = [(k, frequentMeshTermsYear[k]) for k in sorted(frequentMeshTermsYear, key=frequentMeshTermsYear.get, reverse=True)]
    sorted_frequentMeshTermsYear = sorted(frequentMeshTermsYear.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sorted_frequentMeshTermsYear:
        meshTerm = key
        count = value
        with open(ROOT_PATH+os.sep+"transformationMatrix"+os.sep+"semanticallyStableTerms"+os.sep+year, 'a') as the_file:
            the_file.write(meshTerm+"\t"+count)
            the_file.write("\n")


def main():
    frequentMeshTermsYear1 = findMostfrequentWordsInYear(year1)
    print("frequentMeshTermsYear1: ", str(len(frequentMeshTermsYear1)))
    writeToFile(frequentMeshTermsYear1, year1)
    frequentMeshTermsYear2 = findMostfrequentWordsInYear(year2)
    print("frequentMeshTermsYear2: ", str(len(frequentMeshTermsYear2)))
    writeToFile(frequentMeshTermsYear2, year2)

if __name__ == "__main__":
    main()

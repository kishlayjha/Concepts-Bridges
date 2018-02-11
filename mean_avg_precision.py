import numpy as np

matrix1 = np.zeros((6,10), dtype='float64')
matrix2 = np.zeros((6,10), dtype='float64')
matrix3 = np.zeros((6,10), dtype='float64')
matrix4 = np.zeros((6,10), dtype='float64')
matrix5 = np.zeros((6,10), dtype='float64')


def loadMatrix(filePath1, matrix):
    for counter, index in enumerate(open(filePath1)):
        # print(counter)
        splitIndex = index.strip().split("\t")
        counter2 = 0
        # print(len(splitIndex))
        for counter1, val in enumerate(splitIndex):
            # print(val)

            # if counter2>=len(splitIndex)-1:
            #     break

            if counter1 == 0:
                continue
            else:
                # print(counter2)
                matrix[counter][counter2] = val
                # print(matrix1)
                counter2 = counter2+1
    # print(matrix)
    return matrix


def main():
    filePath1 = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files/baselines/KDD/MAP/1"
    filePath2 = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files/baselines/KDD/MAP/2"
    filePath3 = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files/baselines/KDD/MAP/3"
    filePath4 = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files/baselines/KDD/MAP/4"
    filePath5 = "/home/super-machine/PycharmProjects/EmbeddingTransformation/files/baselines/KDD/MAP/5"

    matrixOne = loadMatrix(filePath1, matrix1)
    matrixTwo  = loadMatrix(filePath2, matrix2)
    matrixThree = loadMatrix(filePath3, matrix3)
    matrixFour = loadMatrix(filePath4, matrix4)
    matrixFive = loadMatrix(filePath5, matrix5)

    # print(matrixOne)
    # print(matrixTwo)

    matrixSum1 = np.add(matrixOne, matrixTwo)
    matrixSum2 = np.add(matrixSum1, matrixThree)
    matrixSum3 = np.add(matrixSum2, matrixFour)
    matrixSum4 = np.add(matrixSum3, matrixFive)
    # print(matrixSum)
    print(0.2*matrixSum4)
    # loadMatrix(filePath1)
    # loadMatrix(filePath1)
    # loadMatrix(filePath1)

if __name__ == "__main__":
    main()
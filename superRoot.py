import settings
from class_NearestNeighbour import class_NearestNeighbour
import os
import numpy as np
import main_globalTransformation
import main_neighbours
import anchorPairs
import main_localTransformation
import main_local_and_global
import precision_recall
import spearman_evaluation


def main():

    # main_neighbours.main()
    # print("** Module 1 Complete **")
    # anchorPairs.main()
    # print("** Module 2 Complete **")
    # main_globalTransformation.main()
    # print("** Module 3 Complete **")
    # main_localTransformation.main()
    # print("** Module 4 Complete **")
    # main_local_and_global.main()
    # print("** Module 5 Complete **")

    precision_recall.main()
    print("** Precisions and Recall Values **")


    # spearman_evaluation.main()
    # print("** spearman evaluation ****")



if __name__ == "__main__":
    main()

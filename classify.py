from classifiers.tools import Tools
from classifiers import knn as knn_classifier
from classifiers import Tools
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, getopt


def set_path(argv):
    PATH = ""
    try:
        opts, args = getopt.getopt(argv, "hd:")
    except getopt.GetoptError:
        print("INI_PATH -m MAX -n Normalizer")
    
    for opt, arg in opts:
        if opt == "-h":
            pass
        elif opt in ("-d"):
            DATA_PATH = arg
    return DATA_PATH

def main(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df = df[["cmd", "audio", "adas", "vision", "maxload", "loadstd", "valid"]]
    # print(df.head(5))
    df = df[df.maxload != 0]
    valid, nonvalid = Tools.get_feasibility(df)
    print("\nFeasible: {} Non-Feasible: {}".format(valid, 
    nonvalid))

    scaled_df = Tools.min_max_scaler(df)
    X, y = Tools.get_X_y(scaled_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    krange = range(1, 60)
    knn_classifier.KNeighborsClassifier
    knn = knn_classifier.Knn(X_train, X_test, y_train, y_test)
    knn.set_krange(krange)
    err = knn.best_k()
    knn.plot_best_k(krange, err)

    # knn_30 = knn.train(k=30)
if __name__ == "__main__":
    DATA_PATH = set_path(sys.argv[1:])
    main(DATA_PATH)
import pandas as pd
import matplotlib.pyplot as plt
root_path = "/home/blackstone/PycharmProjects/NIH/AsymmetricDivisions/simulations/"


class Visualizator:
    def __init__(self, path_to_csv):
        self.table = pd.read_csv(path_to_csv, sep="\t")

    def plot(self, x_feature, y_feature):
        plt.plot(self.table[x_feature], self.table[y_feature])
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()


visualizator = Visualizator(path_to_csv=root_path+"data/history.tsv")
visualizator.plot("generation", "n_cells")

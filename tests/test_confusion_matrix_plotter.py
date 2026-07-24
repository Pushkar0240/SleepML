import numpy as np

from src.visualization.confusion_matrix_plotter import ConfusionMatrixPlotter

y_true = np.array([

    0,0,0,1,1,1,0,1,0,1

])

y_pred = np.array([

    0,0,1,1,1,0,0,1,0,1

])

plotter = ConfusionMatrixPlotter()

plotter.plot_and_save(

    y_true,

    y_pred

)

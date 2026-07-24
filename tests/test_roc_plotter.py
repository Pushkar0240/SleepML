import numpy as np

from src.visualization.roc_plotter import ROCPlotter

np.random.seed(42)

y_true = np.random.randint(

    0,

    2,

    500

)

probabilities = np.random.rand(

    500

)

plotter = ROCPlotter()

plotter.add_model(

    y_true,

    probabilities,

    label="CNN1D"

)

plotter.plot_and_save()

print()

print(

    plotter.best_threshold()

)

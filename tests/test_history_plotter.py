from src.visualization.history_plotter import HistoryPlotter

plotter = HistoryPlotter(

    history_file="outputs/history/history.csv"

)

plotter.summary()

plotter.dashboard()

plotter.export_report()

print()

print(plotter.latest_metrics())

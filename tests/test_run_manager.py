from src.experiments import RunManager

manager = RunManager()

print()

print("=" * 60)

print("RUN MANAGER CREATED")

print("=" * 60)

print()

print(

    manager.config.project["name"]

)

print(

    manager.experiment.experiment_id

)

print()

print("Initialization Successful.")

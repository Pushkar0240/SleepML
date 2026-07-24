from src.training.metrics import MetricFactory

metrics = MetricFactory.build()

print()

print("Training Metrics")

print("-" * 40)

for metric in metrics:

    print(metric.name)

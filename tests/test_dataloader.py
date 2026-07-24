from src.training.dataloader import EEGDataLoader

loader = EEGDataLoader(

    batch_size=16

)

dataset = loader.dataset()

for X, y in dataset.take(1):

    print()

    print("Batch")

    print()

    print(X.shape)

    print(y.shape)

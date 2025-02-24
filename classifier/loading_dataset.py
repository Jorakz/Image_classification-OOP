from torchvision import datasets, transforms

def load_mnist():
    # Download MNIST using torchvision and convert to numpy arrays
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    print(f'Dataset Train: x - {len(X_train)}, y - {len(y_train)}')
    print(f'Dataset Train: x - {len(X_test)}, y - {len(y_test)}')
    return X_train, y_train, X_test, y_test

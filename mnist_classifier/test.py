import numpy as np
import torch
from torchvision import datasets, transforms

# Import the classifiers from our modules
from rf import MnistRFClassifier
from nn import MnistNNClassifier
from cnn import MnistCNNClassifier
def load_mnist():
    # Download MNIST dataset using torchvision
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def main():
    X_train, y_train, X_test, y_test = load_mnist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Random Forest ---
    print("=== Random Forest Classifier ===")
    rf_classifier = MnistRFClassifier()
    rf_classifier.train(X_train, y_train)
    evaluate_model(rf_classifier, X_test, y_test)
    
    # --- Feed-Forward Neural Network ---
    print("\n=== Feed-Forward Neural Network ===")
    nn_classifier = MnistNNClassifier(device=device)
    nn_classifier.train(X_train, y_train, epochs=5, batch_size=128)
    evaluate_model(nn_classifier, X_test, y_test)
    
    # --- Convolutional Neural Network ---
    print("\n=== Convolutional Neural Network ===")
    cnn_classifier = MnistCNNClassifier(device=device)
    cnn_classifier.train(X_train, y_train, epochs=5, batch_size=128)
    evaluate_model(cnn_classifier, X_test, y_test)

if __name__ == "__main__":
    main()
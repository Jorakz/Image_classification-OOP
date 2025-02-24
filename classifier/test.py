import numpy as np
import torch
# Import our classifiers
from models.rf import MnistRFClassifier
from models.nn import MnistNNClassifier
from models.cnn import MnistCNNClassifier
# Import plotting functions from result_plot.py
from result_plot import plot_metrics, plot_confusion_matrix
from loading_dataset import load_mnist
from sklearn.metrics import classification_report



def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, predictions))
    return predictions


def main():
    # Visualize the dataset by calling the main() function from analize_dataset.py
    print("Visualizing MNIST dataset:")
    # Load MNIST dataset



    X_train, y_train, X_test, y_test = load_mnist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Random Forest ---
    print("\n=== Random Forest Classifier ===")
    rf_classifier = MnistRFClassifier()
    rf_classifier.train(X_train, y_train)
    rf_predictions = evaluate_model(rf_classifier, X_test, y_test)
    plot_confusion_matrix(y_test, rf_predictions, "Random Forest")

    # --- Feed-Forward Neural Network ---
    print("\n=== Feed-Forward Neural Network ===")
    nn_classifier = MnistNNClassifier(device=device)
    history_nn = nn_classifier.train(X_train, y_train, epochs=10, batch_size=128, X_val=X_test, y_val=y_test)
    nn_predictions = evaluate_model(nn_classifier, X_test, y_test)
    plot_metrics(history_nn, "Feed-Forward NN")
    plot_confusion_matrix(y_test, nn_predictions, "Feed-Forward NN")

    # --- Convolutional Neural Network ---
    print("\n=== Convolutional Neural Network ===")
    cnn_classifier = MnistCNNClassifier(device=device)
    history_cnn = cnn_classifier.train(X_train, y_train, epochs=10, batch_size=128, X_val=X_test, y_val=y_test)
    cnn_predictions = evaluate_model(cnn_classifier, X_test, y_test)
    plot_metrics(history_cnn, "Convolutional NN")
    plot_confusion_matrix(y_test, cnn_predictions, "Convolutional NN")


if __name__ == "__main__":
    main()
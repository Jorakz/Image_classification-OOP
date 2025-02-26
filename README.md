# Image_classificator-OOP

An object-oriented implementation of various machine learning models for MNIST handwritten digit classification. This repository demonstrates how to use different classifier architectures (Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network) with a common interface.

## Project Structure

```
Image_classificator-OOP/
├── interfaces.py            # Abstract base class for MNIST classifiers
├── loading_dataset.py       # Functions to load the MNIST dataset
├── result_plot.py           # Visualization utilities
├── test.py                  # Main script to run all models and compare results
├── models/
│   ├── rf.py                # Random Forest classifier implementation
│   ├── nn.py                # Feed-forward Neural Network implementation
│   └── cnn.py               # Convolutional Neural Network implementation
├── demo.ipynb               # Jupyter notebook for demonstration
└── requirements.txt         # Dependencies for the project
```

## Requirements

The following packages are required to run this project:

```
# Install dependencies
pip install -r requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Image_classificator-OOP.git

# Change to project directory
cd Image_classificator-OOP

# Install requirements
pip install -r requirements.txt
```

## Usage

### Running the Full Test Suite

To train and evaluate all models:

```bash
python test.py
```

This will:
1. Load the MNIST dataset
2. Train each classifier (Random Forest, Feed-Forward NN, CNN)
3. Evaluate each model's performance
4. Generate performance visualizations

### Using the Demo Notebook

For an interactive demonstration:

```bash
jupyter notebook demo.ipynb
```

The notebook provides a step-by-step guide to:
- Loading and exploring the MNIST dataset
- Training each model type
- Comparing performance metrics
- Visualizing results and misclassifications

### Using a Specific Classifier

Each classifier implements the same interface, making them interchangeable:

```python
from models.cnn import MnistCNNClassifier
from loading_dataset import load_mnist

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Create and train a model
classifier = MnistCNNClassifier()
classifier.train(X_train, y_train, epochs=10)

# Make predictions
predictions = classifier.predict(X_test)
```

## Model Architectures

### Random Forest
- Uses scikit-learn's RandomForestClassifier with 100 trees
- Simple but effective baseline model
- Flattens 28×28 images to 784 features

### Feed-Forward Neural Network
- 784 input units (flattened 28×28 images)
- 512 hidden units with ReLU activation
- 20% dropout for regularization
- 10 output units (one per digit)
- Adam optimizer with learning rate 0.001

### Convolutional Neural Network
- Convolutional layer: 32 filters with 3×3 kernel
- ReLU activation
- 2×2 max pooling
- 25% dropout
- Fully connected layers (32×14×14 → 128 → 10)
- Adam optimizer with learning rate 0.001

## Performance

The models generally achieve the following accuracy on the MNIST test set:
- Random Forest: ~96-97%
- Feed-Forward NN: ~97-98%
- CNN: ~98-99%

See the demo notebook for detailed performance metrics and visualizations.

## License

[MIT](LICENSE)

# MNIST Classification with PyTorch

A deep learning project implementing a CNN model for MNIST digit classification with CI/CD pipeline integration.

## Model Architecture

- **Input Layer**: Convolutional layer with BatchNorm and Dropout
- **Feature Extraction**: Multiple conv blocks with ReLU, BatchNorm, and Dropout
- **Pooling**: MaxPool and Global Average Pooling
- **Output**: 10-class classification (digits 0-9)
- **Parameters**: < 100,000 params
- **Key Features**: 
  - Batch Normalization
  - Dropout (p=0.05)
  - Global Average Pooling

## Project Structure 

Tests verify:
1. Parameter count (< 20K)
2. Batch Normalization usage
3. Dropout implementation
4. Global Average Pooling/Fully Connected layer

## CI/CD Pipeline

GitHub Actions workflow automatically:
- Runs all tests
- Trains model
- Verifies saved model loading
- Ensures code quality and functionality

## Performance

- Training Time: ~5-10 minutes on CPU
- Test Accuracy: ~99% on MNIST test set
- Model Size: < 1MB

## License

MIT License

## Author

[Your Name]
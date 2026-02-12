# Forest-Cover-Type-Classification-with-Cosine-Decay-Keras
This project builds and optimizes a multi-class neural network to classify forest cover types using the Covertype dataset (7 classes, 54 features). A balanced subset was created from the original imbalanced dataset to ensure fair model evaluation.

A fixed MLP architecture (64 → 32 → Softmax) was trained using TensorFlow/Keras and systematically tuned across:

Batch sizes (4, 8, 16, 128)

Constant learning rates (1e-5 to 1e-3)

Cosine Decay learning rate schedules

The best-performing configuration (Adam + CosineDecay, initial LR = 5e-3) achieved:

88.1% training accuracy (20 epochs)

84.1% test accuracy

This project demonstrates structured hyperparameter experimentation, reproducible model comparison, and controlled performance optimization on structured tabular data.

# Custom-CNN-on-CIFAR-10
An implementation of a custom convolutional neural network with residual blocks (a simplified ResNet) on the CIFAR‑10 dataset using PyTorch.
This project includes:
- Data Preprocessing & Augmentation: Random cropping and horizontal flipping for robust training.
- Custom CNN with Residual Blocks: A ResNet‑18–style architecture tailored for CIFAR‑10.
- Training Workflow: A full training loop with forward pass, loss computation, backward pass, optimizer update, learning rate scheduling, and periodic evaluation.
- Model Checkpointing: Saving the best-performing model.

Residual Block:
The ResidualBlock class encapsulates two convolutional layers with batch normalization and ReLU activation. A skip connection adds the input (or a downsampled version) to the block's output, which is then activated.

CustomResNet Architecture:
The network starts with an initial convolution, batch normalization, and ReLU.
Four sequential layers are built using the _make_layer method, which stacks a specified number of residual blocks. Some layers downsample the input (via stride and a 1×1 convolution) to reduce spatial dimensions.
An adaptive average pooling layer reduces the feature map to 1×1 before flattening.
A fully connected layer outputs logits for 10 CIFAR‑10 classes.

Data Preprocessing & Augmentation:
For training, random cropping (with padding) and horizontal flipping are applied to improve robustness.
Both training and test sets are normalized using CIFAR‑10’s mean and standard deviation.

Training Workflow:
The training loop processes batches by performing a forward pass, computing cross‑entropy loss, and updating model parameters via backpropagation.
The learning rate scheduler reduces the learning rate every 10 epochs.
After each epoch, the model is evaluated on the test set. The best model (by accuracy) is saved.

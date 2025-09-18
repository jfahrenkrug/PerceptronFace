# PerceptronFace

A Swift implementation of the Rosenblatt Perceptron for binary male/female classification using face images from the UTKFace dataset.

## Background

This implements the classic **Mark I Perceptron** developed by Frank Rosenblatt in 1957 at Cornell. The original Mark I was a room-sized machine with 400 photoreceptors that could learn to classify patterns, including distinguishing male from female faces. It was one of the first neural networks to demonstrate that machines could learn from data.

Our implementation follows the same core algorithm: 400 input features (20×20 pixels), learned weights, and a step activation function that outputs +1 (female) or -1 (male).

## Requirements

- macOS 12.0 or later
- Swift 6.1 or later
- ~220MB disk space for dataset

## Usage

```bash
# Download the UTKFace dataset (~220MB)
swift run PerceptronFace download

# Train with default settings (200 epochs, ages 25-35)
swift run PerceptronFace train

# Train with custom parameters (i.e. 500 epochs, ages 20-40)
swift run PerceptronFace train 500 20-40

# Predict male/female from a face image
swift run PerceptronFace predict /path/to/face.jpg

# Convert image to 20×20 format
swift run PerceptronFace convert /path/to/image.jpg
```

## Dataset

Uses the UTKFace dataset (23,708 face images) from Hugging Face with automatic filtering by age range during training.

## Output Files

- `weights.json` - Trained model weights
- `weights_visualization/` - Visual representation of learned weights
- `processed_images/` - 20×20 processed training images

## Performance

Typical results on ages 25-35: ~85-89% test accuracy, converges within 50-200 epochs.

## Technical Details

- **Algorithm**: Single perceptron with step activation
- **Features**: 400 (20×20 grayscale pixels, normalized 0-1)
- **Learning**: Rate decay (0.99 per epoch), stops on convergence
- **Limitations**: Linear separability only (as noted by Minsky & Papert, 1969)
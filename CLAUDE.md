# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PerceptronFace is a Swift implementation of the Rosenblatt Perceptron for binary gender classification using face images. It uses the UTKFace dataset and processes images to 20x20 grayscale format for training and prediction.

## Build and Run Commands

```bash
# Build the project
swift build

# Download UTKFace dataset (~220MB)
swift run PerceptronFace download

# Train the perceptron (default: 200 epochs, ages 25-35)
swift run PerceptronFace train

# Train with custom parameters
swift run PerceptronFace train 500 20-40

# Make predictions on an image
swift run PerceptronFace predict /path/to/image.jpg

# Convert image to 20x20 processed format
swift run PerceptronFace convert /path/to/image.jpg
```

## Architecture Overview

The codebase is organized into five main components:

### Core Neural Network
- **Perceptron.swift**: Implements the Rosenblatt Perceptron with training, prediction, weight persistence via JSON serialization
- Uses learning rate decay (0.99 per epoch) and step activation function
- Weights are accessible for visualization purposes

### Data Handling
- **FaceDataset.swift**: Contains `FaceImage` struct (pixels, gender label, filename, age) and `FaceDataset` with train/test splitting
- **FaceDatasetLoader.swift**: Downloads UTKFace dataset from Hugging Face, processes UTKFace filename format (age_gender_race_date&time.jpg), handles local image loading
- **ImageProcessor.swift**: Cross-platform image processing using CoreGraphics/ImageIO, resizes to 20x20 grayscale, handles both AppKit (macOS) and UIKit

### Dataset Details
- UTKFace dataset: 23,708 face images with age, gender, race labels
- Downloaded from Hugging Face: `https://huggingface.co/datasets/nlphuji/utk_faces/resolve/main/utk_faces_images.zip`
- Extracts to `utkface_dataset/UTKFace/` directory
- Processes images to 20x20 grayscale (400 features)
- Binary classification: Male (-1) vs Female (+1)

### Key Implementation Notes
- Requires macOS 12+ due to Process API usage for curl/unzip
- Weights saved to `weights.json` in project root
- Processed images saved to `processed_images/` directory
- Weight visualizations saved to `weights_visualization/` directory
- Uses UTKFace filename parsing: `age_gender_race_timestamp.jpg` format

### Critical Bug Fix
The dataset loader handles zip extraction directory naming mismatch by renaming `utk_faces_images/` to `UTKFace/` after extraction.

## File Structure
```
Sources/PerceptronFace/
├── main.swift              # CLI interface and visualization functions
├── Perceptron.swift        # Neural network implementation
├── FaceDataset.swift       # Data structures for images and datasets
├── FaceDatasetLoader.swift # Dataset downloading and loading
└── ImageProcessor.swift   # Image processing utilities
```
//
//  main.swift
//  PerceptronFace
//
//  Created by Johannes Fahrenkrug on 18.09.25.
//  https://springenwerk.com
//

import Foundation

func visualizeImage(_ image: FaceImage, prediction: Int? = nil) {
    print("\n" + String(repeating: "=", count: 42))
    print("Image: \(image.filename)")
    if let age = image.age {
        print("Age: \(age)")
    }
    if let pred = prediction {
        let predLabel = pred == 1 ? "FEMALE" : "MALE"
        let actualLabel = image.isFemale ? "FEMALE" : "MALE"
        print("Predicted: \(predLabel)")
        print("Actual: \(actualLabel)")
        print(prediction == (image.isFemale ? 1 : -1) ? "✓ Correct" : "✗ Incorrect")
    }
    print(String(repeating: "-", count: 42))

    let size = 20
    for y in 0..<size {
        var line = ""
        for x in 0..<size {
            let pixel = image.pixels[y * size + x]
            if pixel < 0.2 {
                line += "  "
            } else if pixel < 0.4 {
                line += ".."
            } else if pixel < 0.6 {
                line += "**"
            } else if pixel < 0.8 {
                line += "##"
            } else {
                line += "██"
            }
        }
        print(line)
    }
    print(String(repeating: "=", count: 42))
}

func visualizeWeights(_ weights: [Double], title: String) {
    print("\n" + String(repeating: "=", count: 42))
    print(title)
    print(String(repeating: "-", count: 42))

    // Normalize weights to 0-1 range for visualization
    let minWeight = weights.min() ?? 0
    let maxWeight = weights.max() ?? 1
    let range = maxWeight - minWeight

    let size = 20
    for y in 0..<size {
        var line = ""
        for x in 0..<size {
            let weight = weights[y * size + x]
            let normalized = range > 0 ? (weight - minWeight) / range : 0.5

            if normalized < 0.2 {
                line += "  "
            } else if normalized < 0.4 {
                line += ".."
            } else if normalized < 0.6 {
                line += "**"
            } else if normalized < 0.8 {
                line += "##"
            } else {
                line += "██"
            }
        }
        print(line)
    }
    print("Min weight: \(String(format: "%.3f", minWeight))")
    print("Max weight: \(String(format: "%.3f", maxWeight))")
    print(String(repeating: "=", count: 42))
}

func printSeparator() {
    print(String(repeating: "=", count: 60))
}

func trainPerceptron(epochs: Int = 200, minAge: Int = 25, maxAge: Int = 35) {
    printSeparator()
    print("ROSENBLATT PERCEPTRON - TRAINING MODE")
    print("Using UTKFace dataset for male/female classification")
    print("Epochs: \(epochs), Age range: \(minAge)-\(maxAge)")
    printSeparator()

    print("\nStep 1: Loading UTKFace dataset...")
    let currentPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

    guard let dataset = FaceDatasetLoader.loadUTKFaceDataset(minAge: minAge, maxAge: maxAge) else {
        print("\nFailed to load dataset. Exiting.")
        return
    }

    print("\nStep 2: Splitting dataset...")
    let (trainData, testData) = dataset.split(trainRatio: 0.8)

    print("Training samples: \(trainData.images.count)")
    print("Test samples: \(testData.images.count)")
    print("Image size: 20x20 pixels (400 features)")
    print("Classes: MALE (-1) vs FEMALE (+1)")

    printSeparator()
    print("\nStep 3: Initializing Perceptron...")
    print("Input size: 400 (20x20 pixels)")
    var perceptron = Perceptron(inputSize: 400)

    // Save and visualize initial weights
    let initialWeights = perceptron.weights
    visualizeWeights(initialWeights, title: "Initial Random Weights (20x20)")

    // Save initial weights as image
    let weightsPath = currentPath.appendingPathComponent("weights_visualization")
    try? FileManager.default.createDirectory(at: weightsPath, withIntermediateDirectories: true)

    // Normalize and save initial weights
    let minInitial = initialWeights.min() ?? 0
    let maxInitial = initialWeights.max() ?? 1
    let rangeInitial = maxInitial - minInitial
    let normalizedInitial = initialWeights.map { rangeInitial > 0 ? ($0 - minInitial) / rangeInitial : 0.5 }
    ImageProcessor.saveImage(pixels: normalizedInitial, size: 20,
                             to: weightsPath.appendingPathComponent("initial_weights.png"))

    printSeparator()
    print("\nStep 4: Training Phase")
    printSeparator()

    // Convert to dataset format: [([Double], Int)]
    let combinedDataset = zip(trainData.features, trainData.labels).map { ($0, $1) }
    perceptron.train(dataset: combinedDataset, epochs: epochs)

    // Visualize and save final weights
    let finalWeights = perceptron.weights
    visualizeWeights(finalWeights, title: "Final Learned Weights (20x20)")

    // Save final weights as image
    let minFinal = finalWeights.min() ?? 0
    let maxFinal = finalWeights.max() ?? 1
    let rangeFinal = maxFinal - minFinal
    let normalizedFinal = finalWeights.map { rangeFinal > 0 ? ($0 - minFinal) / rangeFinal : 0.5 }
    ImageProcessor.saveImage(pixels: normalizedFinal, size: 20,
                             to: weightsPath.appendingPathComponent("final_weights.png"))

    print("\nWeight visualizations saved to: ./weights_visualization/")

    printSeparator()
    print("\nStep 5: Evaluation Phase")
    printSeparator()

    let trainAccuracy = perceptron.evaluate(inputs: trainData.features, labels: trainData.labels)
    let testAccuracy = perceptron.evaluate(inputs: testData.features, labels: testData.labels)

    print("Training Accuracy: \(String(format: "%.2f", trainAccuracy))%")
    print("Test Accuracy: \(String(format: "%.2f", testAccuracy))%")

    printSeparator()
    print("\nStep 6: Sample Predictions")
    printSeparator()

    let samplesToShow = min(3, testData.images.count)
    for i in 0..<samplesToShow {
        let image = testData.images[i]
        let prediction = perceptron.predict(image.pixels)
        visualizeImage(image, prediction: prediction)
    }

    // Save trained weights to JSON
    do {
        try perceptron.saveWeights(to: "weights.json")
        print("\n✓ Weights saved to weights.json")
    } catch {
        print("\n✗ Failed to save weights: \(error)")
    }

    printSeparator()
    print("\nTRAINING COMPLETED!")
    print("\nTo make predictions, use:")
    print("swift run perceptron predict /path/to/image.jpg")
    printSeparator()
}

func predictImage(imagePath: String) {
    printSeparator()
    print("ROSENBLATT PERCEPTRON - PREDICTION MODE")
    printSeparator()

    // Load weights from JSON
    let perceptron: Perceptron
    do {
        perceptron = try Perceptron.loadWeights(from: "weights.json")
        print("✓ Loaded trained weights from weights.json")
    } catch {
        print("✗ Error: Could not load weights.json")
        print("Please run 'swift run perceptron train' first to train the model.")
        return
    }

    print("\nLoading image: \(imagePath)")

    guard let testImage = FaceDatasetLoader.loadLocalImage(at: imagePath) else {
        print("✗ Error: Could not load image at \(imagePath)")
        return
    }

    // Visualize the input image
    print("\nInput image (20x20):")
    visualizeImage(testImage, prediction: nil)

    // Make prediction
    let prediction = perceptron.predict(testImage.pixels)
    let label = prediction == 1 ? "FEMALE" : "MALE"

    printSeparator()
    print("PREDICTION: \(label)")

    if let age = testImage.age {
        print("Detected age from filename: \(age)")
        let actualLabel = testImage.isFemale ? "FEMALE" : "MALE"
        print("Actual label from filename: \(actualLabel)")
        if prediction == (testImage.isFemale ? 1 : -1) {
            print("✓ Prediction is CORRECT")
        } else {
            print("✗ Prediction is INCORRECT")
        }
    }
    printSeparator()
}

func convertImage(imagePath: String) {
    printSeparator()
    print("ROSENBLATT PERCEPTRON - CONVERSION MODE")
    printSeparator()

    print("\nLoading image: \(imagePath)")

    let url = URL(fileURLWithPath: imagePath)

    guard let cgImage = ImageProcessor.loadImage(from: url),
            let pixels = ImageProcessor.resizeAndConvertToGrayscale(image: cgImage, size: 20) else {
        print("Error: Could not load or process image at \(imagePath)")
        return
    }

    // Save the processed 20x20 image
    let filename = url.lastPathComponent
    let processedFilename = "processed_\(filename)"
    let processedURL = url.deletingLastPathComponent().appendingPathComponent(processedFilename)
    ImageProcessor.saveImage(pixels: pixels, size: 20, to: processedURL)
}

func printUsage() {
    print("Usage:")
    print("  swift run PerceptronFace download")
    print("    Download the UTKFace dataset (~220MB)")
    print("")
    print("  swift run PerceptronFace train [epochs] [min_age-max_age]")
    print("    Train the perceptron (defaults: 200 epochs, ages 25-35)")
    print("    Example: swift run PerceptronFace train 500 20-40")
    print("")
    print("  swift run PerceptronFace predict <image_path>")
    print("    Predict gender from an image using trained weights")
    print("    Example: swift run PerceptronFace predict /path/to/face.jpg")
    print("")
    print("  swift run PerceptronFace convert <image_path>")
    print("    Convert image to 20x20 grayscale format")
    print("    Example: swift run PerceptronFace convert /path/to/image.jpg")
}

func main() {
    let args = CommandLine.arguments

    // If no arguments or help requested
    if args.count < 2 || args[1] == "--help" || args[1] == "-h" {
        printUsage()
        return
    }

    let command = args[1]

    switch command {
    case "download":
        let success = FaceDatasetLoader.downloadUTKFaceDataset()
        if !success {
            print("\nDownload failed. Please try again or check your internet connection.")
        }

    case "train":
        var epochs = 200
        var minAge = 25
        var maxAge = 35

        // Parse epochs if provided
        if args.count > 2 {
            if let epochsArg = Int(args[2]) {
                epochs = epochsArg
            }
        }

        // Parse age range if provided
        if args.count > 3 {
            let ageRange = args[3].split(separator: "-")
            if ageRange.count == 2,
               let min = Int(ageRange[0]),
               let max = Int(ageRange[1]) {
                minAge = min
                maxAge = max
            }
        }

        trainPerceptron(epochs: epochs, minAge: minAge, maxAge: maxAge)

    case "predict":
        if args.count < 3 {
            print("Error: Please provide an image path")
            print("Example: swift run PerceptronFace predict /path/to/image.jpg")
            return
        }
        predictImage(imagePath: args[2])

    case "convert":
        if args.count < 3 {
            print("Error: Please provide an image path")
            print("Example: swift run PerceptronFace convert /path/to/image.jpg")
            return
        }
        convertImage(imagePath: args[2])

    default:
        print("Unknown command: \(command)")
        printUsage()
    }
}

main()
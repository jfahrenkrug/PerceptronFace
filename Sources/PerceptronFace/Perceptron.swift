//
//  Perceptron.swift
//  PerceptronFace
//
//  Created by Johannes Fahrenkrug on 18.09.25.
//  https://springenwerk.com
//

import Foundation

struct Perceptron: Codable {
    var weights: [Double]
    var bias: Double

    init(inputSize: Int) {
        self.weights = (0..<inputSize).map { _ in Double.random(in: -0.5...0.5) }
        self.bias = Double.random(in: -0.5...0.5)
    }

    init(from weights: [Double], bias: Double) {
        self.weights = weights
        self.bias = bias
    }

    private func activate(_ value: Double) -> Int {
        return value >= 0 ? 1 : -1
    }

    func predict(_ inputs: [Double]) -> Int {
        let weightedSum = zip(inputs, weights)
            .map { $0 * $1 }
            .reduce(0, +) + bias
        return activate(weightedSum)
    }

    mutating func train(
        dataset: [([Double], Int)],
        learningRate: Double = 0.01,
        epochs: Int
    ) {
        print("Starting training for \(epochs) epochs...")
        print("Initial weights: \(weights.map { String(format: "%.3f", $0) })")
        print("Initial bias: \(String(format: "%.3f", bias))\n")

        var currentLearningRate = learningRate
        let decayRate = 0.99  // Decay learning rate by 1% each epoch

        for epoch in 1...epochs {
            var totalError = 0

            for (input, label) in dataset {
                let prediction = predict(input)

                // Calculate error
                let error = label - prediction

                if error != 0 {
                    totalError += 1

                    // Update weights
                    for i in 0..<weights.count {
                        weights[i] += currentLearningRate * Double(error) * input[i]
                    }

                    // Update bias
                    bias += currentLearningRate * Double(error)
                }
            }

            // Decay learning rate after each epoch
            currentLearningRate *= decayRate

            if epoch % 10 == 0 || epoch == 1 || totalError == 0 {
                let accuracy = Double(dataset.count - totalError) / Double(dataset.count) * 100
                print("Epoch \(epoch): Errors = \(totalError), Accuracy = \(String(format: "%.1f", accuracy))%, LR = \(String(format: "%.6f", currentLearningRate))")

                if totalError == 0 {
                    print("\nConverged at epoch \(epoch)!")
                    break
                }
            }
        }

        print("\nFinal weights: \(weights.map { String(format: "%.3f", $0) })")
        print("Final bias: \(String(format: "%.3f", bias))")
    }

}
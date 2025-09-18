//
//  Perceptron.swift
//  PerceptronFace
//
//  Created by Johannes Fahrenkrug on 18.09.25.
//  https://springenwerk.com
//

import Foundation

struct Perceptron: Codable {
    var weights: [Double]  // Made non-private for visualization
    var bias: Double
    private let learningRate: Double

    init(inputSize: Int, learningRate: Double = 0.01) {
        self.learningRate = learningRate
        self.weights = (0..<inputSize).map { _ in Double.random(in: -0.5...0.5) }
        self.bias = Double.random(in: -0.5...0.5)
    }

    init(from weights: [Double], bias: Double) {
        self.learningRate = 0.01  // Default for loaded weights
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

    mutating func train(inputs: [[Double]], labels: [Int], epochs: Int) {
        print("Starting training for \(epochs) epochs...")
        print("Initial weights: \(weights.map { String(format: "%.3f", $0) })")
        print("Initial bias: \(String(format: "%.3f", bias))\n")

        var currentLearningRate = learningRate
        let decayRate = 0.99  // Decay learning rate by 1% each epoch

        for epoch in 1...epochs {
            var totalError = 0

            for (input, label) in zip(inputs, labels) {
                let prediction = predict(input)
                let error = label - prediction

                if error != 0 {
                    totalError += 1

                    for i in 0..<weights.count {
                        weights[i] += currentLearningRate * Double(error) * input[i]
                    }
                    bias += currentLearningRate * Double(error)
                }
            }

            // Decay learning rate after each epoch
            currentLearningRate *= decayRate

            if epoch % 10 == 0 || epoch == 1 || totalError == 0 {
                let accuracy = Double(inputs.count - totalError) / Double(inputs.count) * 100
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

    func evaluate(inputs: [[Double]], labels: [Int]) -> Double {
        var correct = 0

        for (input, label) in zip(inputs, labels) {
            if predict(input) == label {
                correct += 1
            }
        }

        return Double(correct) / Double(inputs.count) * 100
    }

    func saveWeights(to path: String) throws {
        let weightsData = [
            "weights": weights,
            "bias": bias
        ] as [String: Any]

        let jsonData = try JSONSerialization.data(withJSONObject: weightsData, options: .prettyPrinted)
        let url = URL(fileURLWithPath: path)
        try jsonData.write(to: url)
        print("Weights saved to: \(path)")
    }

    static func loadWeights(from path: String) throws -> Perceptron {
        let url = URL(fileURLWithPath: path)
        let jsonData = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: jsonData, options: []) as! [String: Any]

        guard let weights = json["weights"] as? [Double],
              let bias = json["bias"] as? Double else {
            throw NSError(domain: "Perceptron", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid weights file format"])
        }

        return Perceptron(from: weights, bias: bias)
    }
}
//
//  Perceptron+Extensions.swift
//  PerceptronFace
//
//  Created by Johannes Fahrenkrug on 18.09.25.
//  https://springenwerk.com
//

import Foundation

// MARK: - Evaluation
extension Perceptron {
    func evaluate(inputs: [[Double]], labels: [Int]) -> Double {
        var correct = 0

        for (input, label) in zip(inputs, labels) {
            if predict(input) == label {
                correct += 1
            }
        }

        return Double(correct) / Double(inputs.count) * 100
    }
}

// MARK: - Weight Persistence
extension Perceptron {
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
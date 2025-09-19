import Foundation
import Testing
@testable import PerceptronFace

@Suite("Perceptron")
struct PerceptronTests {
    @Test("Predict returns +1 when weighted sum is non-negative")
    func predictReturnsPositiveLabelWhenWeightedSumIsNonNegative() {
        let perceptron = Perceptron(from: [0.5, -0.25], bias: 0.1)

        #expect(perceptron.predict([0.5, 0.2]) == 1)
    }

    @Test("Predict returns -1 when weighted sum is negative")
    func predictReturnsNegativeLabelWhenWeightedSumIsNegative() {
        let perceptron = Perceptron(from: [-0.5, -0.5], bias: -0.1)

        #expect(perceptron.predict([0.2, 0.3]) == -1)
    }

    @Test("Training corrects misclassified positive example")
    func trainCorrectsMisclassifiedPositiveExample() {
        var perceptron = Perceptron(from: [0.0, 0.0], bias: -0.5)
        let dataset = [([1.0, 1.0], 1)]

        perceptron.train(dataset: dataset, learningRate: 0.5, epochs: 1)

        #expect(perceptron.predict([1.0, 1.0]) == 1)
    }

    @Test("Training corrects misclassified negative example")
    func trainCorrectsMisclassifiedNegativeExample() {
        var perceptron = Perceptron(from: [0.0, 0.0], bias: 0.5)
        let dataset = [([1.0, 1.0], -1)]

        perceptron.train(dataset: dataset, learningRate: 0.5, epochs: 1)

        #expect(perceptron.predict([1.0, 1.0]) == -1)
    }

    @Test("Evaluate returns accuracy percentage")
    func evaluateReturnsAccuracyPercentage() {
        let perceptron = Perceptron(from: [1.0, 1.0], bias: -0.5)
        let inputs: [[Double]] = [[0.1, 0.5], [0.2, 0.1], [1.0, 1.0]]
        let labels = [1, -1, 1]

        #expect(perceptron.evaluate(inputs: inputs, labels: labels) == 100.0)
    }

    @Test("Weights persist to and from disk")
    func saveAndLoadWeightsRoundTripsModelParameters() throws {
        let temporaryURL = FileManager.default.temporaryDirectory.appendingPathComponent("weights-test.json")
        defer { try? FileManager.default.removeItem(at: temporaryURL) }

        let original = Perceptron(from: [0.1, -0.2, 0.3], bias: 0.4)
        try original.saveWeights(to: temporaryURL.path)

        let restored = try Perceptron.loadWeights(from: temporaryURL.path)

        #expect(restored.weights == original.weights)
        #expect(restored.bias == original.bias)
    }
}

import Testing
@testable import PerceptronFace

@Suite("FaceDataset")
struct FaceDatasetTests {
    @Test("Collects all pixel vectors")
    func featuresReturnsAllPixelArrays() {
        let samplePixels1 = [0.1, 0.2, 0.3]
        let samplePixels2 = [0.4, 0.5, 0.6]
        let images = [
            FaceImage(pixels: samplePixels1, isFemale: false, filename: "a", age: 30),
            FaceImage(pixels: samplePixels2, isFemale: true, filename: "b", age: 32)
        ]
        let dataset = FaceDataset(images: images)

        #expect(dataset.features == [samplePixels1, samplePixels2])
    }

    @Test("Maps gender to signed classes")
    func labelsMapsGenderToSignedClasses() {
        let images = [
            FaceImage(pixels: [0.0], isFemale: false, filename: "a", age: 28),
            FaceImage(pixels: [0.0], isFemale: true, filename: "b", age: 29)
        ]
        let dataset = FaceDataset(images: images)

        #expect(dataset.labels == [-1, 1])
    }

    @Test("Splits dataset into expected counts")
    func splitProducesDatasetsWithExpectedCounts() {
        let images = (0..<10).map { index in
            FaceImage(pixels: [Double(index)], isFemale: index.isMultiple(of: 2), filename: "\(index)", age: 26)
        }
        let dataset = FaceDataset(images: images)

        let (train, test) = dataset.split(trainRatio: 0.7)
        let expectedTrainCount = Int(Double(images.count) * 0.7)

        #expect(train.images.count == expectedTrainCount)
        #expect(train.images.count + test.images.count == images.count)
    }
}

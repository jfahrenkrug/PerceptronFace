import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import PerceptronFace

@Suite("ImageProcessor")
struct ImageProcessorTests {
    private func makeGrayscaleImage(width: Int, height: Int) -> CGImage {
        let pixelCount = width * height
        let pixelValues = (0..<pixelCount).map { UInt8(($0 * 255) / max(pixelCount - 1, 1)) }
        let data = Data(pixelValues)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let provider = CGDataProvider(data: data as CFData) else {
            fatalError("Failed to create data provider")
        }
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            fatalError("Failed to create test image")
        }
        return image
    }

    @Test("Produces expected pixel count when resizing")
    func resizeAndConvertToGrayscaleProducesExpectedPixelCount() {
        let image = makeGrayscaleImage(width: 40, height: 40)

        let pixels = ImageProcessor.resizeAndConvertToGrayscale(image: image, size: 20)

        #expect(pixels?.count == 400)
    }

    @Test("Normalizes grayscale values between 0 and 1")
    func resizeAndConvertToGrayscaleNormalizesPixelValuesBetweenZeroAndOne() throws {
        let image = makeGrayscaleImage(width: 40, height: 40)

        let pixels = try #require(ImageProcessor.resizeAndConvertToGrayscale(image: image, size: 20))

        #expect((pixels.min() ?? -1) >= 0.0)
        #expect((pixels.max() ?? 2) <= 1.0)
    }

    @Test("Persists PNG image to disk")
    func saveImageWritesPNGFile() {
        let pixels = Array(repeating: 0.5, count: 400)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("image-test.png")
        defer { try? FileManager.default.removeItem(at: url) }

        ImageProcessor.saveImage(pixels: pixels, size: 20, to: url)

        let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil)
        #expect(imageSource != nil)
    }
}

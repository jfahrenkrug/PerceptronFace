import Foundation
import CoreGraphics
import ImageIO
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

class ImageProcessor {
    static func loadImage(from url: URL) -> CGImage? {
        guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            return nil
        }
        return image
    }

    static func loadImage(from data: Data) -> CGImage? {
        guard let imageSource = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            return nil
        }
        return image
    }

    static func resizeAndConvertToGrayscale(image: CGImage, size: Int) -> [Double]? {
        let width = size
        let height = size

        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return nil
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let pixelData = context.data else {
            return nil
        }

        let data = pixelData.bindMemory(to: UInt8.self, capacity: width * height)
        var pixels: [Double] = []

        for i in 0..<(width * height) {
            pixels.append(Double(data[i]) / 255.0)
        }

        return pixels
    }

    static func saveImage(pixels: [Double], size: Int, to url: URL) {
        let width = size
        let height = size

        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue

        var pixelData = pixels.map { UInt8($0 * 255) }

        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return
        }

        guard let cgImage = context.makeImage() else {
            return
        }

        #if canImport(AppKit)
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
        if let tiffData = nsImage.tiffRepresentation,
           let bitmapRep = NSBitmapImageRep(data: tiffData),
           let pngData = bitmapRep.representation(using: .png, properties: [:]) {
            try? pngData.write(to: url)
        }
        #elseif canImport(UIKit)
        let uiImage = UIImage(cgImage: cgImage)
        if let pngData = uiImage.pngData() {
            try? pngData.write(to: url)
        }
        #endif
    }
}
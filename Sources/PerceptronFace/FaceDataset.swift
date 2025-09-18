//
//  FaceDataset.swift
//  PerceptronFace
//
//  Created by Johannes Fahrenkrug on 18.09.25.
//  https://springenwerk.com
//

import Foundation

struct FaceImage {
    let pixels: [Double]
    let isFemale: Bool
    let filename: String
    let age: Int?
}

struct FaceDataset {
    let images: [FaceImage]
    let imageSize: Int = 20

    var features: [[Double]] {
        return images.map { $0.pixels }
    }

    var labels: [Int] {
        return images.map { $0.isFemale ? 1 : -1 }  // 1 for female, -1 for male
    }

    func split(trainRatio: Double = 0.8) -> (train: FaceDataset, test: FaceDataset) {
        let shuffled = images.shuffled()
        let trainSize = Int(Double(shuffled.count) * trainRatio)

        let trainImages = Array(shuffled[0..<trainSize])
        let testImages = Array(shuffled[trainSize..<shuffled.count])

        return (
            train: FaceDataset(images: trainImages),
            test: FaceDataset(images: testImages)
        )
    }
}
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "PerceptronFace",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(
            name: "PerceptronFace",
            targets: ["PerceptronFace"]
        )
    ],
    targets: [
        .executableTarget(
            name: "PerceptronFace",
            path: "Sources/PerceptronFace"
        )
    ]
)

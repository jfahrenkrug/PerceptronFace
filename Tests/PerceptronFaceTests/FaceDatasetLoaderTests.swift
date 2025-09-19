import Foundation
import Testing
@testable import PerceptronFace

private enum FixtureError: Error {
    case missingFixturesDirectory
    case missingFixture(String)
}

@Suite("FaceDatasetLoader")
struct FaceDatasetLoaderTests {
    private func fixtureURL(named name: String) throws -> URL {
        guard let baseURL = Bundle.module.resourceURL?.appendingPathComponent("Fixtures", isDirectory: true) else {
            throw FixtureError.missingFixturesDirectory
        }
        let url = baseURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FixtureError.missingFixture(name)
        }
        return url
    }

    @Test("Parses female gender from filename")
    func loadLocalImageParsesFemaleGenderFromFilename() throws {
        let url = try fixtureURL(named: "26_1_1_20170116225222631.jpg.chip.jpg")

        let image = FaceDatasetLoader.loadLocalImage(at: url.path)

        #expect(image?.isFemale == true)
    }

    @Test("Parses male gender from filename")
    func loadLocalImageParsesMaleGenderFromFilename() throws {
        let url = try fixtureURL(named: "25_0_0_20170117153733428.jpg.chip.jpg")

        let image = FaceDatasetLoader.loadLocalImage(at: url.path)

        #expect(image?.isFemale == false)
    }

    @Test("Extracts age from filename")
    func loadLocalImageExtractsAgeFromFilename() throws {
        let url = try fixtureURL(named: "29_1_0_20170113012617623.jpg.chip.jpg")

        let image = FaceDatasetLoader.loadLocalImage(at: url.path)

        #expect(image?.age == 29)
    }

    @Test("Produces normalized pixel array")
    func loadLocalImageProducesNormalizedPixelArray() throws {
        let url = try fixtureURL(named: "26_1_1_20170116225222631.jpg.chip.jpg")

        let image = try #require(FaceDatasetLoader.loadLocalImage(at: url.path))

        #expect(image.pixels.count == 400)
        #expect((image.pixels.min() ?? -1) >= 0.0)
        #expect((image.pixels.max() ?? 2) <= 1.0)
    }
}

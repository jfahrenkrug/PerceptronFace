import Foundation

class FaceDatasetLoader {
    static func downloadUTKFaceDataset() -> Bool {
        print("==========================================")
        print("UTKFace Dataset Downloader")
        print("==========================================")
        print("")
        print("This will download the UTKFace dataset (23,708 face images)")
        print("Dataset size: ~220MB compressed")
        print("")

        let currentPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let datasetPath = currentPath.appendingPathComponent("utkface_dataset")
        let utkFacePath = datasetPath.appendingPathComponent("UTKFace")

        // Create dataset directory
        do {
            try FileManager.default.createDirectory(at: datasetPath, withIntermediateDirectories: true)
        } catch {
            print("Error creating dataset directory: \(error)")
            return false
        }

        // Check if dataset already exists
        if FileManager.default.fileExists(atPath: utkFacePath.path) {
            do {
                let files = try FileManager.default.contentsOfDirectory(at: utkFacePath, includingPropertiesForKeys: nil)
                let imageCount = files.filter { $0.pathExtension == "jpg" }.count
                if imageCount > 0 {
                    print("Dataset already exists in \(utkFacePath.path)")
                    print("Found \(imageCount) images")
                    print("Delete the directory to re-download.")
                    return true
                }
            } catch {
                // Directory exists but can't read it, continue with download
            }
        }

        let zipFile = datasetPath.appendingPathComponent("utk_faces_images.zip")
        let downloadURL = "https://huggingface.co/datasets/nlphuji/utk_faces/resolve/main/utk_faces_images.zip?download=true"

        print("Downloading UTKFace dataset...")
        print("From: \(downloadURL)")
        print("This may take a few minutes...")
        print("")

        // Download using curl
        let downloadTask = Process()
        downloadTask.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        downloadTask.arguments = ["-L", "--progress-bar", downloadURL, "-o", zipFile.path]

        do {
            try downloadTask.run()
            downloadTask.waitUntilExit()

            if downloadTask.terminationStatus != 0 {
                print("Error: Download failed with status \(downloadTask.terminationStatus)")
                return false
            }
        } catch {
            print("Error running download: \(error)")
            return false
        }

        // Check if download was successful
        guard FileManager.default.fileExists(atPath: zipFile.path) else {
            print("ERROR: Download failed or file is empty.")
            print("Please try downloading manually from:")
            print(downloadURL)
            return false
        }

        print("")
        print("Download complete. Unzipping...")

        // Unzip the file
        let unzipTask = Process()
        unzipTask.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        unzipTask.arguments = ["-q", zipFile.path]
        unzipTask.currentDirectoryURL = datasetPath

        do {
            try unzipTask.run()
            unzipTask.waitUntilExit()

            if unzipTask.terminationStatus != 0 {
                print("Error: Unzip failed with status \(unzipTask.terminationStatus)")
                return false
            }
        } catch {
            print("Error running unzip: \(error)")
            return false
        }

        // Clean up the zip file
        try? FileManager.default.removeItem(at: zipFile)

        // Check if extraction was successful and rename directory if needed
        let extractedPath = datasetPath.appendingPathComponent("utk_faces_images")

        if FileManager.default.fileExists(atPath: extractedPath.path) {
            // Rename the extracted directory to UTKFace for consistency
            do {
                if FileManager.default.fileExists(atPath: utkFacePath.path) {
                    try FileManager.default.removeItem(at: utkFacePath)
                }
                try FileManager.default.moveItem(at: extractedPath, to: utkFacePath)
            } catch {
                print("Error renaming extracted directory: \(error)")
                return false
            }
        }

        if FileManager.default.fileExists(atPath: utkFacePath.path) {
            do {
                let files = try FileManager.default.contentsOfDirectory(at: utkFacePath, includingPropertiesForKeys: nil)
                let imageCount = files.filter { $0.pathExtension == "jpg" }.count

                print("")
                print("==========================================")
                print("Success! Dataset extracted to: \(utkFacePath.path)")
                print("Total images found: \(imageCount)")
                print("==========================================")

                // Show sample filenames
                print("")
                print("Sample image filenames:")
                let sampleFiles = files.filter { $0.pathExtension == "jpg" }.prefix(5)
                for file in sampleFiles {
                    print(file.lastPathComponent)
                }

                print("")
                print("Dataset is ready for use!")
                return true
            } catch {
                print("Error reading extracted directory: \(error)")
                return false
            }
        } else {
            print("")
            print("ERROR: Extraction failed. UTKFace directory not found.")
            return false
        }
    }
    static func loadUTKFaceDataset(minAge: Int = 25, maxAge: Int = 35) -> FaceDataset? {
        print("\nLoading UTKFace dataset...")
        print("This dataset contains 23,708 faces labeled with age, gender, and race")
        print("Filtering to ages \(minAge)-\(maxAge)")

        var images: [FaceImage] = []

        let currentPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let datasetPath = currentPath.appendingPathComponent("utkface_dataset")

        // Create directory for processed images
        let processedPath = currentPath.appendingPathComponent("processed_images")
        try? FileManager.default.createDirectory(at: processedPath, withIntermediateDirectories: true)

        try? FileManager.default.createDirectory(at: datasetPath, withIntermediateDirectories: true)

        // Load real UTKFace images from local directory
        let localUTKPath = datasetPath.appendingPathComponent("UTKFace")

        if !FileManager.default.fileExists(atPath: localUTKPath.path) {
            print("\nERROR: UTKFace dataset not found!")
            print("Please run: swift run PerceptronFace download")
            print("This will download the dataset (~220MB)")
            return nil
        }

        print("Found UTKFace dataset at: \(localUTKPath.path)")

        do {
            let files = try FileManager.default.contentsOfDirectory(at: localUTKPath,
                                                                   includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "jpg" }
                .shuffled()

            print("Processing \(files.count) UTKFace images...")

            var processedCount = 0
            var skippedCount = 0
            for (_, fileURL) in files.enumerated() {
                let filename = fileURL.lastPathComponent
                let components = filename.replacingOccurrences(of: ".jpg", with: "")
                                         .replacingOccurrences(of: ".chip", with: "")
                                         .split(separator: "_")

                guard components.count >= 3,
                      let age = Int(components[0]),
                      let gender = Int(components[1]) else {
                    continue
                }

                // Skip ages outside specified range before any processing
                if age < minAge || age > maxAge {
                    skippedCount += 1
                    continue
                }

                if let cgImage = ImageProcessor.loadImage(from: fileURL),
                   let pixels = ImageProcessor.resizeAndConvertToGrayscale(image: cgImage, size: 20) {

                    // Save the processed 20x20 image
                    let processedFilename = "processed_\(filename)"
                    let processedURL = processedPath.appendingPathComponent(processedFilename)
                    ImageProcessor.saveImage(pixels: pixels, size: 20, to: processedURL)

                    images.append(FaceImage(
                        pixels: pixels,
                        isFemale: gender == 1,
                        filename: filename,
                        age: age
                    ))
                    processedCount += 1

                    if processedCount % 1000 == 0 {
                        print("Processed \(processedCount)/\(files.count) images...")
                    }
                }
            }

            print("Successfully processed \(processedCount) UTKFace images!")
            print("Skipped \(skippedCount) images outside age range \(minAge)-\(maxAge)")

        } catch {
            print("ERROR: Could not read UTKFace directory: \(error)")
            print("Please run: swift run PerceptronFace download")
            return nil
        }

        if images.isEmpty {
            print("\nERROR: No valid images found in dataset")
            return nil
        }

        print("\n" + String(repeating: "=", count: 60))
        print("Dataset loaded: \(images.count) images")
        print("Males: \(images.filter { !$0.isFemale }.count)")
        print("Females: \(images.filter { $0.isFemale }.count)")
        let ages = images.compactMap({ $0.age })
        if !ages.isEmpty {
            let avgAge = ages.reduce(0, +) / ages.count
            print("Average age: \(avgAge)")
        }
        print("\nProcessed 20x20 images saved to: ./processed_images/")
        print(String(repeating: "=", count: 60))

        return FaceDataset(images: images)
    }

    static func loadLocalImage(at path: String) -> FaceImage? {
        let url = URL(fileURLWithPath: path)

        guard let cgImage = ImageProcessor.loadImage(from: url),
              let pixels = ImageProcessor.resizeAndConvertToGrayscale(image: cgImage, size: 20) else {
            print("Error: Could not load or process image at \(path)")
            return nil
        }

        // Try to parse gender from filename if it follows UTKFace format
        let filename = url.lastPathComponent
        var isFemale = false  // Default to male
        var age: Int? = nil

        let components = filename.replacingOccurrences(of: ".jpg", with: "")
                                 .replacingOccurrences(of: ".png", with: "")
                                 .split(separator: "_")
        if components.count >= 2,
           let parsedAge = Int(components[0]),
           let gender = Int(components[1]) {
            age = parsedAge
            isFemale = gender == 1
        }

        return FaceImage(
            pixels: pixels,
            isFemale: isFemale,
            filename: url.lastPathComponent,
            age: age
        )
    }
}
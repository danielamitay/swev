import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import Swev

private func encodedImage(width: Int, height: Int, rgba: [UInt8], jpeg: Bool = false, orientation: Int = 1) throws -> Data {
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let output = NSMutableData()
    let destination = CGImageDestinationCreateWithData(output, (jpeg ? "public.jpeg" : "public.png") as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation, kCGImageDestinationLossyCompressionQuality: 1] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else { throw SwevError.invalidRequest("Cannot encode test image") }
    return output as Data
}

@Test func imageFitAndAlpha() throws {
    let config = ImagePreprocessing(width: 4, height: 4, resize: "fit-nearest", background: [255, 255, 255], tokenSequence: "image")
    let data = try encodedImage(width: 2, height: 1, rgba: [255, 0, 0, 255, 0, 255, 0, 255])
    let pixels = try config.pixels(.init(data: data, contentType: "image/png"))
    func rgb(_ x: Int, _ y: Int) -> [Float] { (0..<3).map { pixels[(y * 4 + x) * 3 + $0].floatValue } }
    #expect(rgb(0, 0) == [1, 1, 1])
    #expect(rgb(0, 1) == [1, 0, 0])
    #expect(rgb(3, 2) == [0, 1, 0])
    #expect(rgb(0, 3) == [1, 1, 1])
    let transparent = try encodedImage(width: 1, height: 1, rgba: [255, 0, 0, 0])
    let alpha = try config.pixels(.init(data: transparent, contentType: "image/png"))
    #expect((0..<alpha.count).allSatisfy { alpha[$0].floatValue == 1 })
    let absent = try config.pixels(nil)
    #expect((0..<absent.count).allSatisfy { absent[$0].floatValue == 0 })
    #expect(throws: SwevError.self) { try config.pixels(.init(data: data, contentType: "image/jpeg")) }
    #expect(throws: SwevError.self) { try config.pixels(.init(data: Data([1,2,3]), contentType: "image/png")) }
}

@Test func imageEXIFOrientation() throws {
    let colors: [[UInt8]] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]]
    var rgba: [UInt8] = []
    for y in 0..<32 { for x in 0..<64 { rgba += colors[(y < 16 ? 0 : 2) + (x < 32 ? 0 : 1)] } }
    let permutations = [[0,1,2,3], [1,0,3,2], [3,2,1,0], [2,3,0,1], [0,2,1,3], [2,0,3,1], [3,1,2,0], [1,3,0,2]]
    for orientation in 1...8 {
        let data = try encodedImage(width: 64, height: 32, rgba: rgba, jpeg: true, orientation: orientation)
        let width = orientation >= 5 ? 32 : 64, height = orientation >= 5 ? 64 : 32
        let config = ImagePreprocessing(width: width, height: height, resize: "fit-nearest", background: [255, 255, 255], tokenSequence: "image")
        let pixels = try config.pixels(.init(data: data, contentType: "image/jpeg"))
        for quadrant in 0..<4 {
            let x = quadrant % 2 == 0 ? width / 4 : width * 3 / 4
            let y = quadrant < 2 ? height / 4 : height * 3 / 4
            for channel in 0..<3 {
                let expected = Float(colors[permutations[orientation - 1][quadrant]][channel]) / 255
                #expect(abs(pixels[(y * width + x) * 3 + channel].floatValue - expected) < 0.03)
            }
        }
    }
}

private struct ImageManifest: Decodable {
    let model: String
    let preprocessing: String
    let cases: String
    let textReference: String
    let report: String?
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["SWEV_IMAGE_TEST_MANIFEST"] != nil))
func imageModelReferenceParity() async throws {
    let location = ProcessInfo.processInfo.environment["SWEV_IMAGE_TEST_MANIFEST"]!
    let manifest = try JSONDecoder().decode(ImageManifest.self, from: Data(contentsOf: URL(fileURLWithPath: location)))
    let pre = try JSONDecoder().decode(ModelAssets.Preprocessing.self, from: Data(contentsOf: URL(fileURLWithPath: manifest.preprocessing)))
    let config = try #require(pre.image)
    let model = try await SwevModel.load(from: URL(fileURLWithPath: manifest.model), configuration: .init(computeUnits: .cpuOnly))
    let caseURL = URL(fileURLWithPath: manifest.cases)
    let cases = try JSONValue.parse(Data(contentsOf: caseURL))
    guard case .array(let rows) = cases else { Issue.record("Invalid image references"); return }
    var reports: [[String: Any]] = []
    for row in rows {
        guard case .string(let imageName) = row.member("image"), case .string(let pixelName) = row.member("pixels"),
              case .array(let expected) = row.member("probabilities") else { Issue.record("Invalid image fixture"); return }
        let image = ImageInput(data: try Data(contentsOf: caseURL.deletingLastPathComponent().appendingPathComponent(imageName)), contentType: "image/png")
        let pixels = try config.pixels(image)
        let reference = try Data(contentsOf: caseURL.deletingLastPathComponent().appendingPathComponent(pixelName))
        guard reference.count == pixels.count * 4 else { Issue.record("Invalid pixel reference size"); return }
        let pixelError = reference.withUnsafeBytes { bytes -> Float in
            var maximum: Float = 0
            for i in 0..<pixels.count { maximum = max(maximum, abs(pixels[i].floatValue - bytes.loadUnaligned(fromByteOffset: i * 4, as: Float.self))) }
            return maximum
        }
        #expect(pixelError <= 1e-7, "Image preprocessing mismatch: \(imageName)")
        let decoded = try DecisionCodec.decodeRequest(Data(try row.member("request")!.jsonString().utf8))
        let clock = ContinuousClock(); let start = clock.now
        let response = try await model.predict(state: decoded.state, questions: decoded.questions, images: [image])
        let duration = start.duration(to: clock.now).components
        let probabilities: [Double]
        switch response.answers[0] {
        case .noul(_, let value): probabilities = [1 - value.noul, value.noul]
        case .choice(_, let value): probabilities = value.probabilities.map(\.probability)
        case .score(_, let value): probabilities = value.probabilities
        }
        let target = expected.map { if case .number(let value) = $0 { return value }; return Double.nan }
        #expect(probabilities.count == target.count)
        #expect(zip(probabilities, target).allSatisfy { abs($0 - $1) < 1e-4 }, "Image model mismatch: \(imageName)")
        reports.append(["image": imageName, "seconds": Double(duration.seconds) + Double(duration.attoseconds) / 1e18,
                        "pixel_error": pixelError, "probabilities": probabilities,
                        "response": try JSONSerialization.jsonObject(with: DecisionCodec.encodeResponse(response))])
    }
    guard case .array(let textRows) = try JSONValue.parse(Data(contentsOf: URL(fileURLWithPath: manifest.textReference))) else { Issue.record("Invalid text references"); return }
    for row in textRows {
        let request = try DecisionCodec.decodeRequest(Data(try row.member("request")!.jsonString().utf8))
        let response = try await model.predict(request)
        let actual: [Double]
        switch response.answers[0] {
        case .noul(_, let value): actual = [1 - value.noul, value.noul]
        case .choice(_, let value): actual = value.probabilities.map(\.probability)
        case .score(_, let value): actual = value.probabilities
        }
        guard case .array(let expected) = row.member("probabilities") else { Issue.record("Invalid text probabilities"); return }
        #expect(actual.count == expected.count)
        #expect(zip(actual, expected).allSatisfy { value, expected in
            guard case .number(let target) = expected else { return false }; return abs(value - target) < 1e-4
        })
    }
    let question = Question.noul(id: "q", instructions: "Is the image red?")
    let invalid = ImageInput(data: Data([1, 2, 3]), contentType: "image/png")
    await #expect(throws: SwevError.self) { try await model.predict(state: "test", questions: [question], images: [invalid]) }
    await #expect(throws: SwevError.self) { try await model.predict(state: "test", questions: [question], images: [invalid, invalid]) }
    if let report = manifest.report {
        try JSONSerialization.data(withJSONObject: reports, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: report))
    }
}

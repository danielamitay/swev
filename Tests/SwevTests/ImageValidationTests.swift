import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import Swev

private func encodedImage(type: String, width: Int = 2, height: Int = 3) throws -> Data {
    let context = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    let image = try #require(context.makeImage())
    let data = NSMutableData()
    let destination = try #require(CGImageDestinationCreateWithData(data, type as CFString, 1, nil))
    CGImageDestinationAddImage(destination, image, nil)
    #expect(CGImageDestinationFinalize(destination))
    return data as Data
}

@Test func imageValidationAcceptsMatchingPNGAndJPEG() throws {
    for (type, mime) in [("public.png", "image/png"), ("public.jpeg", "image/jpeg")] {
        let image = try ImageValidation.image(.init(data: encodedImage(type: type), contentType: mime))
        #expect(image.extent.width == 2)
        #expect(image.extent.height == 3)
    }
}

@Test func imageValidationRejectsMismatchedAndUnsupportedFormats() throws {
    for (type, mime) in [("public.jpeg", "image/png"), ("public.png", "image/jpeg"),
                         ("public.tiff", "image/png"), ("public.tiff", "image/tiff")] {
        let input = ImageInput(data: try encodedImage(type: type), contentType: mime)
        #expect(throws: SwevError.invalidRequest("Invalid PNG/JPEG image or mismatched content type")) {
            try ImageValidation.image(input)
        }
    }
}

@Test func imageValidationRejectsMalformedAndOversizedData() {
    for data in [Data(), Data("not an image".utf8)] {
        #expect(throws: SwevError.invalidRequest("Invalid PNG/JPEG image or mismatched content type")) {
            try ImageValidation.image(.init(data: data, contentType: "image/png"))
        }
    }
    #expect(throws: SwevError.resourceLimit) {
        try ImageValidation.image(.init(data: Data(count: 32 * 1024 * 1024 + 1), contentType: "image/png"))
    }
}

@Test func imageValidationRejectsLargeDimensionsFromSmallEncodedFiles() throws {
    // Valid, highly compressible PNGs exercise both side and total-pixel limits.
    for (width, height) in [(8193, 1), (1, 8193), (4097, 4096)] {
        let data = try encodedImage(type: "public.png", width: width, height: height)
        #expect(data.count < 1024 * 1024)
        #expect(throws: SwevError.resourceLimit, "dimensions: \(width)x\(height)") {
            try ImageValidation.image(.init(data: data, contentType: "image/png"))
        }
    }
}

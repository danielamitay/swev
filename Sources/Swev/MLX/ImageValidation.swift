import CoreImage
import Foundation
import ImageIO

/// Checks the input contract before creating an image for the runtime's processor.
/// Reads metadata without decoding pixels; resizing and normalization remain runtime-owned.
enum ImageValidation {
    static func image(_ input: ImageInput) throws -> CIImage {
        guard input.data.count <= 32 * 1024 * 1024 else { throw SwevError.resourceLimit }
        let types = ["image/png": "public.png", "image/jpeg": "public.jpeg"]
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let expectedType = types[input.contentType],
              let source = CGImageSourceCreateWithData(input.data as CFData, options),
              CGImageSourceGetType(source) as String? == expectedType,
              CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw SwevError.invalidRequest("Invalid PNG/JPEG image or mismatched content type")
        }
        // Check each dimension first so multiplication cannot overflow.
        guard (1...8192).contains(width), (1...8192).contains(height),
              width * height <= 16_777_216 else { throw SwevError.resourceLimit }
        guard let image = CIImage(data: input.data, options: [.applyOrientationProperty: true]) else {
            throw SwevError.invalidRequest("Cannot decode PNG/JPEG image")
        }
        return image
    }
}

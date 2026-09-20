import CoreGraphics
import CoreML
import Foundation
import ImageIO

struct ImagePreprocessing: Decodable {
    let width: Int
    let height: Int
    let resize: String
    let background: [Int]
    let tokenSequence: String

    func validate() throws {
        guard (1...1024).contains(width), (1...1024).contains(height), resize == "fit-nearest",
              background.count == 3, background.allSatisfy({ (0...255).contains($0) }),
              !tokenSequence.isEmpty, tokenSequence.utf8.count <= 16384 else { throw SwevError.invalidMetadata }
    }

    /// RGB float values in NHWC order. Missing images produce an unused zero tensor.
    func pixels(_ input: ImageInput?) throws -> MLMultiArray {
        try validate()
        let result = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width), 3], dataType: .float32)
        let output = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        output.initialize(repeating: 0, count: result.count)
        guard let input else { return result }
        guard !input.data.isEmpty, input.data.count <= 32 * 1024 * 1024 else { throw SwevError.resourceLimit }
        let types = ["image/png": "public.png", "image/jpeg": "public.jpeg"]
        guard let expectedType = types[input.contentType],
              let source = CGImageSourceCreateWithData(input.data as CFData, nil),
              CGImageSourceGetCount(source) == 1,
              CGImageSourceGetType(source) as String? == expectedType,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let sourceWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let sourceHeight = properties[kCGImagePropertyPixelHeight] as? Int else { throw SwevError.invalidRequest("Invalid PNG or JPEG image") }
        guard (1...8192).contains(sourceWidth), (1...8192).contains(sourceHeight),
              sourceWidth * sourceHeight <= 16_777_216 else { throw SwevError.resourceLimit }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        guard (1...8).contains(orientation),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
              image.width == sourceWidth, image.height == sourceHeight,
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { throw SwevError.invalidRequest("Cannot decode image") }
        var rgba = [UInt8](repeating: 0, count: sourceWidth * sourceHeight * 4)
        try rgba.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(data: bytes.baseAddress, width: sourceWidth, height: sourceHeight,
                                          bitsPerComponent: 8, bytesPerRow: sourceWidth * 4, space: colorSpace,
                                          bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue) else { throw SwevError.resourceLimit }
            context.setFillColor(red: CGFloat(background[0]) / 255, green: CGFloat(background[1]) / 255,
                                 blue: CGFloat(background[2]) / 255, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))
        }
        let orientedWidth = orientation >= 5 ? sourceHeight : sourceWidth
        let orientedHeight = orientation >= 5 ? sourceWidth : sourceHeight
        let scale = min(Double(width) / Double(orientedWidth), Double(height) / Double(orientedHeight))
        let targetWidth = max(1, Int((Double(orientedWidth) * scale).rounded()))
        let targetHeight = max(1, Int((Double(orientedHeight) * scale).rounded()))
        let left = (width - targetWidth) / 2, top = (height - targetHeight) / 2
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 3
                guard x >= left, x < left + targetWidth, y >= top, y < top + targetHeight else {
                    for channel in 0..<3 { output[offset + channel] = Float(background[channel]) / 255 }
                    continue
                }
                let ox = min(orientedWidth - 1, (2 * (x - left) + 1) * orientedWidth / (2 * targetWidth))
                let oy = min(orientedHeight - 1, (2 * (y - top) + 1) * orientedHeight / (2 * targetHeight))
                let sx: Int, sy: Int
                switch orientation {
                case 2: (sx, sy) = (sourceWidth - 1 - ox, oy)
                case 3: (sx, sy) = (sourceWidth - 1 - ox, sourceHeight - 1 - oy)
                case 4: (sx, sy) = (ox, sourceHeight - 1 - oy)
                case 5: (sx, sy) = (oy, ox)
                case 6: (sx, sy) = (oy, sourceHeight - 1 - ox)
                case 7: (sx, sy) = (sourceWidth - 1 - oy, sourceHeight - 1 - ox)
                case 8: (sx, sy) = (sourceWidth - 1 - oy, ox)
                default: (sx, sy) = (ox, oy)
                }
                let sourceOffset = (sy * sourceWidth + sx) * 4
                for channel in 0..<3 { output[offset + channel] = Float(rgba[sourceOffset + channel]) / 255 }
            }
        }
        return result
    }
}

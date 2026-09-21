import Foundation
import CoreFoundation
import MLXLMCommon
import MLXVLM

/// Isolated compatibility for split processor metadata in mlx-swift-lm 3.31.4.
/// All image processing stays in MLX; registries and checkpoint files are never mutated.
enum ProcessorCompatibility {
    static func factory(directory: URL) throws -> VLMModelFactory {
        guard let data = try configuration(directory: directory) else { return .shared }
        let registry = ProcessorTypeRegistry(creators: [
            "Idefics3Processor": { _, tokenizer in
                SmolVLMProcessor(try JSONDecoder().decode(SmolVLMProcessorConfiguration.self, from: data), tokenizer: tokenizer)
            },
            "SmolVLMProcessor": { _, tokenizer in
                SmolVLMProcessor(try JSONDecoder().decode(SmolVLMProcessorConfiguration.self, from: data), tokenizer: tokenizer)
            },
        ])
        return VLMModelFactory(typeRegistry: VLMTypeRegistry.shared,
            processorRegistry: registry, modelRegistry: VLMRegistry.shared)
    }

    static func configuration(directory: URL) throws -> Data? {
        func read(_ filename: String) throws -> [String: Any] {
            let url = directory.appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw SwevError.invalidModelAsset
            }
            return object
        }
        let processor = try read("processor_config.json")
        let preprocessor = try read("preprocessor_config.json")
        // Preserve the runtime's preprocessor precedence, filling fields absent from that file.
        var merged = processor.merging(preprocessor) { _, imageValue in imageValue }
        guard let type = merged["processor_class"] as? String,
              ["Idefics3Processor", "SmolVLMProcessor"].contains(type),
              merged["max_image_size"] != nil else { return nil }
        let model = try read("config.json")
        let lengths = [processor["image_seq_len"], preprocessor["image_seq_len"], model["image_seq_len"]].compactMap { $0 }
        let validLengths = lengths.compactMap { value -> Int? in
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                  let count = Int(number.stringValue), count > 0 else { return nil }
            return count
        }
        guard validLengths.count == lengths.count, Set(validLengths).count <= 1 else {
            throw SwevError.invalidRequest("Conflicting or invalid image_seq_len in model/processor configuration")
        }
        // Require the asset's declared count instead of the runtime's implicit 64-token default.
        guard let length = validLengths.first else {
            throw SwevError.invalidRequest("Image processor must declare image_seq_len")
        }
        merged["image_seq_len"] = length
        merged["video_sampling"] = merged["video_sampling"] ?? ["fps": 1, "max_frames": 20]
        return try JSONSerialization.data(withJSONObject: merged)
    }
}

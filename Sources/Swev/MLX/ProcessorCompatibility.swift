import Foundation
import MLXLMCommon
import MLXVLM

/// Compatibility for mlx-swift-lm 3.31.4: small Idefics3 checkpoints declare the older processor
/// name but require the runtime's SmolVLM chat/image processor. No pixels are processed by Swev.
/// Keep this shim isolated so it can be removed when the upstream registry handles these assets.
enum ProcessorCompatibility {
    static let install: Task<Void, Never> = Task {
        await VLMProcessorTypeRegistry.shared.registerProcessorType("Idefics3Processor") { data, tokenizer in
            guard var configuration = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw SwevError.invalidModelAsset
            }
            guard configuration["max_image_size"] != nil else {
                return Idefics3Processor(try JSONDecoder().decode(Idefics3ProcessorConfiguration.self, from: data), tokenizer: tokenizer)
            }
            // Image-only checkpoints omit this required runtime field; no video input is exposed.
            configuration["video_sampling"] = configuration["video_sampling"] ?? ["fps": 1, "max_frames": 20]
            let decoded = try JSONDecoder().decode(SmolVLMProcessorConfiguration.self,
                from: JSONSerialization.data(withJSONObject: configuration))
            return SmolVLMProcessor(decoded, tokenizer: tokenizer)
        }
    }
}

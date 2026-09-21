import Foundation
import CoreFoundation

/// Reads declared capabilities without allocating model weights or inferring anything from its name.
struct ModelConfigurationInspection {
    let modelType: String
    let maxContextTokens: Int

    init(data: Data, contextLimit: Int? = nil) throws {
        guard let configuration = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = configuration["model_type"] as? String, !modelType.isEmpty else {
            throw SwevError.invalidRequest("Model configuration must declare model_type")
        }
        let text = configuration["text_config"] as? [String: Any] ?? configuration
        let declared = text["max_position_embeddings"] ?? configuration["max_position_embeddings"]
        let declaredLimit: Int?
        if let declared {
            guard let number = declared as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  let limit = Int(number.stringValue), limit > 0 else {
                throw SwevError.invalidRequest("max_position_embeddings must be a positive integer")
            }
            declaredLimit = limit
        } else {
            declaredLimit = nil
        }
        if let contextLimit {
            guard contextLimit > 0, declaredLimit.map({ contextLimit <= $0 }) ?? true else {
                throw SwevError.invalidRequest("maxContextTokens must be positive and cannot exceed the declared context limit")
            }
            maxContextTokens = contextLimit
        } else if let declaredLimit {
            maxContextTokens = declaredLimit
        } else {
            throw SwevError.invalidRequest("Model omits max_position_embeddings; provide maxContextTokens using the model's documented context limit")
        }
        self.modelType = modelType
    }
}

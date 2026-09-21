import CoreML
import CryptoKit
import Foundation

struct ModelAssets {
    struct Preprocessing: Decodable {
        let sequenceLength: Int
        let sequenceBuckets: [Int]?
        let optionCapacity: Int
        let recipe: TextRecipe
        let tensors: String
        var image: ImagePreprocessing? = nil
    }
    struct Postprocessing: Decodable {
        let temperatures: [Double]
        let temperaturesByOptions: [String: Double]
        let choiceConfidence: ConfidenceMethod
        let scoreConfidence: ConfidenceMethod
    }
    struct Asset: Decodable { let key: String; let bytes: Int; let sha256: String }
    struct Feature: Codable, Equatable { let shape: [Int]; let dtype: String; var enumeratedShapes: [[Int]]? = nil }
    struct Signatures: Codable, Equatable { let inputs: [String: Feature]; let outputs: [String: Feature] }
    let descriptor: CoreMLModelDescriptor
    let adapter: TextAdapter
    let tensors: String
    let sequenceBuckets: [Int]
    let image: ImagePreprocessing?
    let postprocessing: Postprocessing

    init(model: MLModel, metadata supplied: [String: String]? = nil) throws {
        try self.init(description: model.modelDescription, metadata: supplied)
    }

    init(description: MLModelDescription, metadata supplied: [String: String]? = nil) throws {
        let metadata = supplied ?? (description.metadata[.creatorDefinedKey] as? [String: String] ?? [:])
        func read<T: Decodable>(_ key: String, _: T.Type) throws -> T {
            guard let text = metadata[key] else { throw SwevError.missingMetadata(key: key) }
            guard text.utf8.count <= 1_048_576 else { throw SwevError.resourceLimit }
            do { return try JSONDecoder().decode(T.self, from: Data(text.utf8)) }
            catch { throw SwevError.invalidMetadata }
        }
        descriptor = try CoreMLModelDescriptor.read(metadata: metadata)
        guard ["text-decision-v1", "vision-decision-v1", "routed-vision-decision-v1"].contains(descriptor.execution.profile) else { throw SwevError.unsupportedProfile(descriptor.execution.profile) }
        let pre = try read("swev.preprocessing", Preprocessing.self)
        guard descriptor.execution.inputAdapter == "text-recipe-v1", ["masked-options", "causal-pointer", "causal-labels"].contains(pre.tensors) else { throw SwevError.unsupportedProfile(descriptor.execution.inputAdapter) }
        let routed = descriptor.execution.profile == "routed-vision-decision-v1"
        guard routed == (metadata["swev.text-model"] != nil) else { throw SwevError.invalidMetadata }
        let isVision = descriptor.execution.profile == "vision-decision-v1" || routed
        guard isVision == (pre.image != nil), !isVision || pre.tensors == "causal-labels" else { throw SwevError.invalidMetadata }
        try pre.image?.validate()
        image = pre.image
        let limits = descriptor.capabilities.limits
        guard descriptor.capabilities.modalities == (isVision ? ["text", "image"] : ["text"]), Set(descriptor.capabilities.questionTypes) == Set(QuestionType.allCases),
              limits.maxQuestionsPerRequest <= 64, (8...(descriptor.contractVersion == "2.0" ? 4096 : 2048)).contains(pre.sequenceLength), (2...32).contains(pre.optionCapacity),
              (pre.sequenceLength == limits.maxSequenceTokens || (routed && descriptor.contractVersion == "2.0" && pre.sequenceLength < limits.maxSequenceTokens && limits.maxSequenceTokens <= 4096)), pre.optionCapacity == limits.maxOptionsPerQuestion else { throw SwevError.invalidMetadata }
        let buckets = pre.sequenceBuckets ?? [pre.sequenceLength]
        guard !buckets.isEmpty, buckets.count <= 16, buckets == Array(Set(buckets)).sorted(),
              buckets.allSatisfy({ (8...pre.sequenceLength).contains($0) }), buckets.last == pre.sequenceLength,
              pre.sequenceBuckets == nil || descriptor.contractVersion == "2.0" else { throw SwevError.invalidMetadata }
        sequenceBuckets = buckets
        let signatures = try read("swev.signatures", Signatures.self)
        func feature(_ shape: [Int], _ dtype: String = "int32") -> Feature { .init(shape: shape, dtype: dtype) }
        let l = buckets[0], k = pre.optionCapacity
        var expected = ["input_ids": feature([1, l]), "option_indices": feature([1, k])]
        if pre.tensors == "masked-options" {
            expected["token_mask"] = feature([1, l]); expected["option_mask"] = feature([1, k]); expected["question_type"] = feature([1])
        } else {
            expected["position_ids"] = feature([1, l]); expected["decision_indices"] = feature([1]); expected["attention_bias"] = feature([1, 1, l, l], "float32")
        }
        if let image = pre.image { expected["image_pixels"] = feature([1, image.height, image.width, 3], "float32") }
        if buckets.count > 1 {
            for name in ["input_ids", "position_ids", "token_mask", "attention_bias"] where expected[name] != nil {
                expected[name]!.enumeratedShapes = buckets.map { name == "attention_bias" ? [1, 1, $0, $0] : [1, $0] }
            }
        }
        let required = Signatures(inputs: expected, outputs: ["option_logits": feature([1, k], "float32")])
        guard signatures == required else { throw SwevError.signatureMismatch }
        func actual(_ features: [String: MLFeatureDescription]) throws -> [String: Feature] {
            try features.mapValues { value in
                guard !value.isOptional, value.type == .multiArray, let constraint = value.multiArrayConstraint else { throw SwevError.signatureMismatch }
                let dtype: String
                switch constraint.dataType {
                case .int32: dtype = "int32"
                case .float32: dtype = "float32"
                default: throw SwevError.signatureMismatch
                }
                var result = feature(constraint.shape.map(\.intValue), dtype)
                let shapes = constraint.shapeConstraint.enumeratedShapes.map { $0.map(\.intValue) }
                if shapes.count > 1 { result.enumeratedShapes = shapes }
                return result
            }
        }
        guard try actual(description.inputDescriptionsByName) == required.inputs,
              try actual(description.outputDescriptionsByName) == required.outputs else { throw SwevError.signatureMismatch }
        let index = try read("swev.tokenizer.asset-index", [String: Asset].self)
        guard !index.isEmpty, index.count <= 16, let tokenizerAsset = index["tokenizer.json"] else { throw SwevError.unsupportedTokenizer }
        var totalBytes = 0
        for asset in index.values {
            guard let payload = metadata[asset.key] else { throw SwevError.missingMetadata(key: asset.key) }
            let data = Data(payload.utf8); totalBytes += data.count
            guard totalBytes <= 32 * 1024 * 1024, data.count == asset.bytes,
                  SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == asset.sha256 else { throw SwevError.metadataIntegrityFailure }
        }
        let tokenizer = try BPETokenizer(data: Data(metadata[tokenizerAsset.key]!.utf8))
        adapter = try TextAdapter(tokenizer: tokenizer, length: pre.sequenceLength, optionCapacity: k, recipe: pre.recipe)
        guard (pre.tensors == "causal-labels") == (pre.recipe.candidateTokens != nil) else { throw SwevError.invalidMetadata }
        guard adapter.imageSlots == (isVision ? 1 : 0) else { throw SwevError.invalidMetadata }
        if let image = pre.image {
            guard try tokenizer.encode(image.tokenSequence).count < l else { throw SwevError.invalidMetadata }
        }
        tensors = pre.tensors
        postprocessing = try read("swev.postprocessing", Postprocessing.self)
        guard postprocessing.temperatures.count == 3,
              (postprocessing.temperatures + Array(postprocessing.temperaturesByOptions.values)).allSatisfy({ $0.isFinite && $0 > 0 }) else { throw SwevError.invalidMetadata }
    }

    func temperature(_ question: Question) -> Double {
        let bucket = question.optionCount == 2 ? "2" : question.optionCount <= 5 ? "3-5" : question.optionCount <= 10 ? "6-10" : "11+"
        let type = question.type == .choice ? 0 : question.type == .score ? 1 : 2
        return postprocessing.temperaturesByOptions[question.type.rawValue + ":" + bucket] ?? postprocessing.temperatures[type]
    }
}

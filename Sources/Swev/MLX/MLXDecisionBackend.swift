import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The runtime owns all model-specific processing. Only evaluated candidate values leave its lock.
final class MLXDecisionBackend: Sendable {
    let id: String
    let revision: String
    let maxContextTokens: Int
    let supportsImages: Bool
    private let container: ModelContainer

    private init(container: ModelContainer, id: String, revision: String, supportsImages: Bool, maxContextTokens: Int) {
        self.container = container
        self.id = id
        self.revision = revision
        self.supportsImages = supportsImages
        self.maxContextTokens = maxContextTokens
    }

    static func load(hf id: String, revision: String) async throws -> MLXDecisionBackend {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw SwevError.invalidRequest("Expected a Hugging Face model ID in owner/name form")
        }
        return try await load(configuration: .init(id: id, revision: revision), id: id, revision: revision, local: false)
    }

    static func load(url: URL) async throws -> MLXDecisionBackend {
        var isDirectory: ObjCBool = false
        guard url.isFileURL,
              FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
              !["mlpackage", "mlmodelc"].contains(url.pathExtension) else { throw SwevError.invalidModelAsset }
        return try await load(configuration: .init(directory: url), id: url.path, revision: "local", local: true)
    }

    private static func load(configuration: ModelConfiguration, id: String, revision: String,
                             local: Bool) async throws -> MLXDecisionBackend {
        try Task.checkCancellation()
        await ProcessorCompatibility.install.value
        let container: ModelContainer
        do {
            container = try await VLMModelFactory.shared.loadContainer(from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(), configuration: configuration)
        } catch ModelFactoryError.unsupportedModelType {
            container = try await LLMModelFactory.shared.loadContainer(from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(), configuration: configuration)
        }
        let loadedConfiguration = await container.configuration
        guard case .directory(let directory) = loadedConfiguration.id,
              let config = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any] else {
            throw SwevError.invalidModelAsset
        }
        let textConfig = config["text_config"] as? [String: Any] ?? config
        guard let contextLimit = textConfig["max_position_embeddings"] as? Int, contextLimit > 0 else {
            throw SwevError.invalidRequest("Model must declare max_position_embeddings for bounded decision input")
        }
        let vision = await container.perform { context in context.model is any VLMModel }
        try Task.checkCancellation()
        let snapshot = directory.lastPathComponent
        let resolvedRevision = snapshot.count == 40 && snapshot.allSatisfy(\.isHexDigit) ? snapshot : revision
        return .init(container: container, id: id, revision: local ? "local" : resolvedRevision,
                     supportsImages: vision, maxContextTokens: contextLimit)
    }

    /// Each question gets a fresh cache and one logical prefill; no token is sampled or generated.
    /// The context bound includes runtime-inserted image tokens. Overlong input is never truncated.
    func predict(_ request: DecisionRequest) async throws -> DecisionResponse {
        try request.validate()
        guard request.images.isEmpty || supportsImages else { throw SwevError.unsupportedModality }
        guard request.images.count <= 1 else { throw SwevError.invalidRequest("At most one image is supported") }
        return try await container.perform { context in
            try Task.checkCancellation()
            let images: [UserInput.Image] = try request.images.map { image in
                guard image.data.count <= 32 * 1024 * 1024 else { throw SwevError.resourceLimit }
                guard ["image/png", "image/jpeg"].contains(image.contentType),
                      let decoded = CIImage(data: image.data, options: [.applyOrientationProperty: true]) else {
                    throw SwevError.invalidRequest("Invalid PNG/JPEG image")
                }
                return .ciImage(decoded)
            }
            var answers: [Answer] = []
            var inputTokens = 0
            for question in request.questions {
                try Task.checkCancellation()
                let prompt = try DecisionPrompt(state: request.state, question: question)
                let candidates = try prompt.candidateTokenIDs { context.tokenizer.encode(text: $0) }
                var input = try await context.processor.prepare(input: UserInput(
                    chat: [.user(prompt.text, images: images)], additionalContext: ["enable_thinking": false]))
                // Continue the runtime's assistant prefix with a deterministic decision prefix.
                // This is still one prefill; no answer token is generated or decoded.
                let suffix = MLXArray(context.tokenizer.encode(text: DecisionPrompt.answerPrefix))
                    .reshaped(input.text.tokens.ndim == 2 ? [1, -1] : [-1])
                let tokens = concatenated([input.text.tokens, suffix], axis: -1)
                input = LMInput(text: .init(tokens: tokens, mask: ones(like: tokens)), image: input.image)
                let count = input.text.tokens.size
                guard count > 0, count <= self.maxContextTokens else { throw SwevError.contextOverflow }
                try Task.checkCancellation()
                let cache = context.model.newCache(parameters: nil)
                let logits: MLXArray
                switch try context.model.prepare(input, cache: cache, windowSize: 512) {
                case .tokens(let remaining):
                    logits = context.model(remaining, cache: cache, state: nil).logits
                case .logits(let output):
                    logits = output.logits
                }
                guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0,
                      candidates.allSatisfy({ $0 < logits.dim(2) }) else { throw SwevError.inferenceFailed }
                let selected = logits[0, -1, MLXArray(candidates)].asType(.float32)
                eval(selected)
                try Task.checkCancellation()
                answers.append(try Postprocessing.answer(question: question,
                    logits: selected.asArray(Float.self).map(Double.init)))
                inputTokens += count
            }
            return DecisionResponse(modelID: self.id, modelRevision: self.revision, answers: answers,
                usage: .init(inputTokens: inputTokens, outputTokens: 0), metadata: request.metadata)
        }
    }
}

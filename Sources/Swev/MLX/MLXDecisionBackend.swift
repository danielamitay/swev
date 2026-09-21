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

    static func load(hf id: String, revision: String, contextLimit: Int? = nil) async throws -> MLXDecisionBackend {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw SwevError.invalidRequest("Expected a Hugging Face model ID in owner/name form")
        }
        return try await load(configuration: .init(id: id, revision: revision), id: id, revision: revision, local: false, contextLimit: contextLimit)
    }

    static func load(url: URL, contextLimit: Int? = nil) async throws -> MLXDecisionBackend {
        var isDirectory: ObjCBool = false
        guard url.isFileURL,
              FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
              !["mlpackage", "mlmodelc"].contains(url.pathExtension) else { throw SwevError.invalidModelAsset }
        return try await load(configuration: .init(directory: url), id: url.path, revision: "local", local: true, contextLimit: contextLimit)
    }

    private static func load(configuration: ModelConfiguration, id: String, revision: String,
                             local: Bool, contextLimit: Int?) async throws -> MLXDecisionBackend {
        try Task.checkCancellation()
        let directory: URL
        let weightConfiguration: ModelConfiguration
        switch configuration.id {
        case .directory(let url):
            directory = url
            weightConfiguration = configuration
        case .id(let modelID, let modelRevision):
            // Inspect metadata first so unsupported architectures never trigger weight downloads.
            directory = try await #hubDownloader().download(id: modelID, revision: modelRevision,
                matching: ["config.json"], useLatest: false, progressHandler: { _ in })
            // Bind subsequent downloads to the inspected snapshot if the downloader exposes it.
            let snapshot = directory.lastPathComponent
            weightConfiguration = snapshot.count == 40 && snapshot.allSatisfy(\.isHexDigit)
                ? .init(id: modelID, revision: snapshot) : configuration
        }
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw SwevError.invalidRequest("This runtime requires config.json with a supported model_type; custom decision-head checkpoints need a dedicated backend")
        }
        let inspection = try ModelConfigurationInspection(data: Data(contentsOf: configURL), contextLimit: contextLimit)
        let useVision = await VLMTypeRegistry.shared.contains(inspection.modelType)
        let useLanguage = await LLMTypeRegistry.shared.contains(inspection.modelType)
        guard useVision || useLanguage else {
            throw SwevError.invalidRequest("Unsupported model_type: \(inspection.modelType). A matching runtime implementation is required")
        }
        let resolved = try await resolve(configuration: weightConfiguration, from: #hubDownloader(),
            useLatest: false, progressHandler: { _ in })
        let container: ModelContainer
        if useVision {
            let factory = try ProcessorCompatibility.factory(directory: resolved.modelDirectory)
            container = ModelContainer(context: try await factory._load(configuration: resolved,
                tokenizerLoader: TokenizerCompatibility()))
        } else {
            container = ModelContainer(context: try await LLMModelFactory.shared._load(configuration: resolved,
                tokenizerLoader: TokenizerCompatibility()))
        }
        let vision = await container.perform { context in context.model is any VLMModel }
        try Task.checkCancellation()
        let snapshot = resolved.modelDirectory.lastPathComponent
        let resolvedRevision = snapshot.count == 40 && snapshot.allSatisfy(\.isHexDigit) ? snapshot : revision
        return .init(container: container, id: id, revision: local ? "local" : resolvedRevision,
                     supportsImages: vision, maxContextTokens: inspection.maxContextTokens)
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

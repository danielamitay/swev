import Foundation
import HuggingFace
import MLX
import MLXDecisionModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The runtime owns all model-specific processing. Only evaluated candidate values leave its lock.
final class MLXDecisionBackend: DecisionRuntime {
    let id: String
    let revision: String
    let maxContextTokens: Int
    let supportsImages: Bool
    private let container: ModelContainer
    private let continuation: AssistantContinuation

    private init(container: ModelContainer, id: String, revision: String, supportsImages: Bool, maxContextTokens: Int, continuation: AssistantContinuation) {
        self.container = container
        self.continuation = continuation
        self.id = id
        self.revision = revision
        self.supportsImages = supportsImages
        self.maxContextTokens = maxContextTokens
    }

    static func load(hf id: String, revision: String, contextLimit: Int? = nil) async throws -> any DecisionRuntime {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw SwevError.invalidRequest("Expected a Hugging Face model ID in owner/name form")
        }
        return try await load(configuration: .init(id: id, revision: revision), id: id, revision: revision, local: false, contextLimit: contextLimit)
    }

    static func load(url: URL, contextLimit: Int? = nil) async throws -> any DecisionRuntime {
        var isDirectory: ObjCBool = false
        guard url.isFileURL,
              FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else { throw SwevError.invalidModelAsset }
        return try await load(configuration: .init(directory: url), id: url.path, revision: "local", local: true, contextLimit: contextLimit)
    }

    private static func load(configuration: ModelConfiguration, id: String, revision: String,
                             local: Bool, contextLimit: Int?) async throws -> any DecisionRuntime {
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
                matching: ["config.json", "mlx_config.json", "rl_agent_config.json", "encoder/config.json"], useLatest: false, progressHandler: { _ in })
            // Bind subsequent downloads to the inspected snapshot if the downloader exposes it.
            let snapshot = directory.lastPathComponent
            weightConfiguration = snapshot.count == 40 && snapshot.allSatisfy(\.isHexDigit)
                ? .init(id: modelID, revision: snapshot) : configuration
        }
        if !FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path),
           FileManager.default.fileExists(atPath: directory.appendingPathComponent("mlx_config.json").path) {
            _ = try EncoderDecisionBackend.configuration(directory: directory, limit: contextLimit)
            let modelDirectory: URL
            switch weightConfiguration.id {
            case .directory(let url): modelDirectory = url
            case .id(let repo, let commit):
                modelDirectory = try await #hubDownloader().download(id: repo, revision: commit,
                    matching: ["mlx_config.json", "rl_agent_config.json", "encoder/config.json", "tokenizer/*", "model.safetensors"],
                    useLatest: false, progressHandler: { _ in })
            }
            return try await EncoderDecisionBackend.load(directory: modelDirectory, id: id,
                revision: local ? "local" : modelDirectory.lastPathComponent, contextLimit: contextLimit)
        }
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw SwevError.invalidRequest("This runtime requires config.json with a supported model_type; custom decision-head checkpoints need a dedicated backend")
        }
        let inspection = try ModelConfigurationInspection(data: Data(contentsOf: configURL), contextLimit: contextLimit)
        let rotated = inspection.modelType == RotatedQuantization.modelType
        let registeredVision = await VLMTypeRegistry.shared.contains(inspection.modelType)
        let useVision = rotated || registeredVision
        let useLanguage = await LLMTypeRegistry.shared.contains(inspection.modelType)
        guard useVision || useLanguage else {
            throw SwevError.invalidRequest("Unsupported model_type: \(inspection.modelType). A matching runtime implementation is required")
        }
        let rotatedFactory = try rotated ? RotatedQuantization.factory(configuration: Data(contentsOf: configURL)) : nil
        let resolved = try await resolve(configuration: weightConfiguration, from: #hubDownloader(),
            useLatest: false, progressHandler: { _ in })
        let container: ModelContainer
        if useVision {
            let factory = try rotatedFactory ?? ProcessorCompatibility.factory(directory: resolved.modelDirectory)
            container = ModelContainer(context: try await factory._load(configuration: resolved,
                tokenizerLoader: TokenizerCompatibility()))
        } else {
            container = ModelContainer(context: try await LLMModelFactory.shared._load(configuration: resolved,
                tokenizerLoader: TokenizerCompatibility()))
        }
        if rotated { try await container.perform { try RotatedQuantization.validate($0.model) } }
        let vision = await container.perform { context in context.model is any VLMModel }
        let continuation = try await container.perform { context in
            try AssistantContinuation(encode: { context.tokenizer.encode(text: $0, addSpecialTokens: false) },
                decode: { context.tokenizer.decode(tokenIds: $0) }, renderAssistant: {
                    let tokens = try context.tokenizer.applyChatTemplate(messages: [
                        ["role": "user", "content": "Ready?"],
                        ["role": "assistant", "content": AssistantContinuation.probe]
                    ], tools: nil, additionalContext: ["enable_thinking": false])
                    return context.tokenizer.decode(tokenIds: tokens)
                })
        }
        try Task.checkCancellation()
        let snapshot = resolved.modelDirectory.lastPathComponent
        let resolvedRevision = snapshot.count == 40 && snapshot.allSatisfy(\.isHexDigit) ? snapshot : revision
        return MLXDecisionBackend(container: container, id: id, revision: local ? "local" : resolvedRevision,
                     supportsImages: vision, maxContextTokens: inspection.maxContextTokens, continuation: continuation)
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
                .ciImage(try ImageValidation.image(image))
            }
            var answers: [Answer] = []
            var inputTokens = 0
            for question in request.questions {
                try Task.checkCancellation()
                let prompt = try DecisionPrompt(state: request.state, question: question)
                let candidates = try prompt.candidateTokenIDs { context.tokenizer.encode(text: $0, addSpecialTokens: false) }
                var input = try await context.processor.prepare(input: UserInput(
                    chat: [.user(prompt.text, images: images)], additionalContext: ["enable_thinking": false]))
                // Continue the runtime's assistant prefix with a deterministic decision prefix.
                // This is still one prefill; no answer token is generated or decoded.
                let tailCount = min(self.continuation.tailCount, input.text.tokens.size)
                let tail = tailCount == 0 ? [] : input.text.tokens.reshaped([-1])[(input.text.tokens.size - tailCount)...].asArray(Int.self)
                let suffix = MLXArray(try self.continuation.tokens(after: tail))
                    .reshaped(input.text.tokens.ndim == 2 ? [1, -1] : [-1])
                let tokens = concatenated([input.text.tokens, suffix], axis: -1)
                input = LMInput(text: .init(tokens: tokens, mask: ones(like: tokens)), image: input.image)
                let count = input.text.tokens.size
                guard count > 0, count <= self.maxContextTokens else { throw SwevError.contextOverflow }
                try Task.checkCancellation()
                let cache = try context.model.newCache(parameters: nil)
                let logits: MLXArray
                switch try context.model.prepare(input, cache: cache, state: nil, prefill: .init(stepSize: 512, chunking: .remainder)) {
                case .tokens(let remaining):
                    logits = context.model(try remaining.batchedForDecisionForward(), cache: cache, state: nil).logits
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

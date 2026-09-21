@preconcurrency import CoreML
import Foundation

/// Core ML devices allowed for execution; availability and performance depend on the model and host.
public enum ComputeUnits: Sendable {
    case all, cpuOnly, cpuAndGPU, cpuAndNeuralEngine

    var coreML: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

/// Execution devices and bounded admission for one resident model instance.
public struct RuntimeConfiguration: Sendable {
    /// Defaults to all available devices. CPU-only is useful for reproducible validation.
    public var computeUnits: ComputeUnits
    /// Maximum admitted requests, including the active request; valid range is 1...64.
    /// Excess requests fail with `SwevError.queueFull` rather than waiting unboundedly.
    public var maxPendingRequests: Int

    public init(computeUnits: ComputeUnits = .all, maxPendingRequests: Int = 8) {
        self.computeUnits = computeUnits
        self.maxPendingRequests = maxPendingRequests
    }
}

/// Modalities, question types, and limits declared by a validated model package.
public struct ModelCapabilities: Decodable, Sendable {
    public let modalities: [String]
    public let questionTypes: [QuestionType]
    public let limits: Limits
    public var supportsImages: Bool { modalities.contains("image") }

    public struct Limits: Decodable, Sendable {
        public let maxQuestionsPerRequest: Int
        public let maxOptionsPerQuestion: Int
        /// Maximum across routes. A particular text/image route or field can have a lower limit.
        public let maxSequenceTokens: Int
    }
}

/// Package identity and execution contract, available after successful loading.
public struct ModelDescriptor: Decodable, Sendable {
    /// Swev metadata schema version; distinct from the export version and source revision.
    public let contractVersion: String
    public let modelVersion: String
    public let revision: String?
    public let id: String
    public let architecture: String
    public let capabilities: ModelCapabilities
    public let execution: Execution

    public struct Execution: Decodable, Sendable {
        public let profile: String
        public let inputAdapter: String
    }

    static func read(metadata: [String: String]) throws -> ModelDescriptor {
        guard let json = metadata["swev.config"] else { throw SwevError.missingMetadata(key: "swev.config") }
        guard json.utf8.count <= 1_048_576 else { throw SwevError.resourceLimit }
        struct Version: Decodable { let contractVersion: String }
        let version: Version
        do { version = try JSONDecoder().decode(Version.self, from: Data(json.utf8)) }
        catch { throw SwevError.invalidMetadata }
        guard ["1.0", "2.0"].contains(version.contractVersion) else {
            throw SwevError.unsupportedContractVersion(version.contractVersion)
        }
        let descriptor: ModelDescriptor
        do { descriptor = try JSONDecoder().decode(Self.self, from: Data(json.utf8)) }
        catch { throw SwevError.invalidMetadata }
        let limits = descriptor.capabilities.limits
        guard !descriptor.id.isEmpty, !descriptor.modelVersion.isEmpty,
              !descriptor.execution.profile.isEmpty, !descriptor.execution.inputAdapter.isEmpty,
              limits.maxQuestionsPerRequest > 0, limits.maxOptionsPerQuestion >= 2,
              limits.maxSequenceTokens > 0 else { throw SwevError.invalidMetadata }
        return descriptor
    }
}

/// A resident Core ML model with serialized, bounded inference and native text preprocessing.
public actor SwevModel {
    public nonisolated let descriptor: ModelDescriptor
    private var model: MLModel?
    private let compiledURL: URL
    private let computeUnits: ComputeUnits
    private let assets: ModelAssets
    private let textRuntime: (model: MLModel, assets: ModelAssets)?
    private var alternateRuntime: (text: Bool, length: Int, model: MLModel)?
    private let ownedTextCompiledURL: URL?
    private let ownedCompiledURL: URL?
    private nonisolated let admission: RequestAdmission

    private init(contentsOf url: URL, configuration: RuntimeConfiguration) async throws {
        try Task.checkCancellation()
        let needsCompilation = url.pathExtension != "mlmodelc"
        let compiled = needsCompilation ? try await MLModel.compileModel(at: url) : url
        var loaded = false
        var textCompiled: URL?
        defer { if !loaded, let textCompiled { try? FileManager.default.removeItem(at: textCompiled) } }
        defer { if needsCompilation && !loaded { try? FileManager.default.removeItem(at: compiled) } }
        ownedCompiledURL = needsCompilation ? compiled : nil
        compiledURL = compiled
        computeUnits = configuration.computeUnits
        try Task.checkCancellation()
        let config = MLModelConfiguration()
        config.computeUnits = configuration.computeUnits.coreML
        let description = try await MLModelAsset(url: compiled).modelDescription
        assets = try ModelAssets(description: description)
        descriptor = assets.descriptor
        if let (textModel, textAssets, textURL) = try BundledTextModel.load(metadata: description.metadata[.creatorDefinedKey] as? [String: String] ?? [:], compiledURL: compiled, configuration: config) {
            textCompiled = textURL
            textRuntime = (textModel, textAssets)
            model = nil
        } else {
            textRuntime = nil
            model = try MLModel(contentsOf: compiled, configuration: config)
        }
        ownedTextCompiledURL = textCompiled
        admission = RequestAdmission(limit: configuration.maxPendingRequests)
        try Task.checkCancellation()
        loaded = true
    }

    deinit {
        if let ownedTextCompiledURL { try? FileManager.default.removeItem(at: ownedTextCompiledURL) }
        if let ownedCompiledURL { try? FileManager.default.removeItem(at: ownedCompiledURL) }
    }

    /// Loads a local `.mlpackage`, `.mlmodel`, or `.mlmodelc` with the Swev contract.
    /// Source assets are compiled off the main actor; temporary compiled files live with this instance.
    /// Keep the returned model resident for repeated predictions. This overload does not download assets.
    /// - Throws: File/Core ML errors or `SwevError` for unsupported or inconsistent metadata,
    ///   signatures, tokenizers, or configuration. Cancellation throws `CancellationError`.
    public static func load(from modelURL: URL, configuration: RuntimeConfiguration = .init()) async throws -> SwevModel {
        try Task.checkCancellation()
        guard modelURL.isFileURL,
              ["mlpackage", "mlmodel", "mlmodelc"].contains(modelURL.pathExtension),
              FileManager.default.fileExists(atPath: modelURL.path) else { throw SwevError.invalidModelAsset }
        guard (1...64).contains(configuration.maxPendingRequests) else { throw SwevError.invalidRequest("Pending request limit must be 1–64") }
        let task = Task.detached { try await SwevModel(contentsOf: modelURL, configuration: configuration) }
        return try await withTaskCancellationHandler {
            let model = try await task.value
            try Task.checkCancellation()
            return model
        } onCancel: { task.cancel() }
    }

    /// Evaluates ordered questions against shared state and an optional PNG/JPEG image.
    /// Equivalent to creating a `DecisionRequest` and calling `predict(_:)`.
    public nonisolated func predict(state: JSONValue, questions: [Question], images: [ImageInput] = [],
                                    metadata: RequestMetadata? = nil) async throws -> DecisionResponse {
        try await predict(.init(state: state, questions: questions, images: images, metadata: metadata))
    }

    /// Runs native preprocessing and serialized Core ML inference off the main actor.
    /// Questions are independent; oversized input is rejected, not truncated. No partial response is returned.
    /// Cancellation is checked between stages and questions; an active device call must finish first.
    /// - Throws: `SwevError` for invalid requests, capacity limits, unsupported images, or a full queue;
    ///   underlying image/Core ML errors can also propagate. Cancellation throws `CancellationError`.
    public nonisolated func predict(_ request: DecisionRequest) async throws -> DecisionResponse {
        try Task.checkCancellation()
        try admission.acquire()
        defer { admission.release() }
        return try await perform(request)
    }

    private func perform(_ request: DecisionRequest) throws -> DecisionResponse {
        // Actor executors do not guarantee an Objective-C pool per request.
        // Release Core ML temporaries and evicted model instances promptly.
        try autoreleasepool { try performPrediction(request) }
    }

    private func performPrediction(_ request: DecisionRequest) throws -> DecisionResponse {
        try Task.checkCancellation()
        guard request.images.isEmpty || assets.image != nil else { throw SwevError.unsupportedModality }
        guard request.images.count <= 1 else { throw SwevError.invalidRequest("At most one image is supported") }
        guard request.questions.count <= descriptor.capabilities.limits.maxQuestionsPerRequest else { throw SwevError.resourceLimit }
        try request.validate()
        let model: MLModel
        let assets: ModelAssets
        if request.images.isEmpty, let textRuntime {
            model = textRuntime.model
            assets = textRuntime.assets
        } else {
            if self.model == nil {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits.coreML
                self.model = try MLModel(contentsOf: compiledURL, configuration: config)
            }
            model = self.model!
            assets = self.assets
        }
        let imagePixels = try assets.image.map { try $0.pixels(request.images.first) }
        let imageTokens = request.images.isEmpty ? nil : assets.image?.tokenSequence
        var answers: [Answer] = []
        var tokens = 0
        for question in request.questions {
            try Task.checkCancellation()
            let encoded = try assets.adapter.encode(state: request.state, question: question, imageTokens: imageTokens)
            let features = try inputs(encoded, imagePixels: imagePixels, assets: assets)
            try Task.checkCancellation()
            let length = assets.sequenceBuckets.first(where: { $0 >= encoded.count })!
            let selected: MLModel
            if length == assets.sequenceBuckets[0] {
                selected = model
            } else {
                let text = request.images.isEmpty && textRuntime != nil
                if alternateRuntime?.text != text || alternateRuntime?.length != length {
                    // Keep the shortest route resident and at most one longer route.
                    // Each instance sees only one shape: some Core ML graphs cannot
                    // safely resize cached intermediate buffers between predictions.
                    alternateRuntime = nil
                    let config = MLModelConfiguration()
                    config.computeUnits = computeUnits.coreML
                    let url = text ? ownedTextCompiledURL! : compiledURL
                    alternateRuntime = (text, length, try MLModel(contentsOf: url, configuration: config))
                }
                selected = alternateRuntime!.model
            }
            let output = try selected.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            try Task.checkCancellation()
            guard let array = output.featureValue(for: "option_logits")?.multiArrayValue,
                  array.count == assets.adapter.optionCapacity else { throw SwevError.inferenceFailed }
            let logits = (0..<question.optionCount).map { array[$0].doubleValue }
            let method = question.type == .score ? assets.postprocessing.scoreConfidence : assets.postprocessing.choiceConfidence
            var answer = try Postprocessing.answer(question: question, logits: logits, temperature: assets.temperature(question), method: method)
            if assets.adapter.recipe.scoreLegend == "indented", case .score(let id, let score) = answer,
               case .score(_, _, let levels) = question {
                answer = .score(id: id, .init(score: score.score, modalLevel: score.modalLevel, probabilities: score.probabilities,
                    legend: try levels.map { try $0.indentedText() }, confidence: score.confidence))
            }
            answers.append(answer)
            tokens += encoded.count
        }
        try Task.checkCancellation()
        return .init(modelID: descriptor.id, modelRevision: descriptor.revision ?? descriptor.modelVersion,
                     answers: answers, usage: .init(inputTokens: tokens, outputTokens: 0), metadata: request.metadata)
    }

    private func inputs(_ row: EncodedQuestion, imagePixels: MLMultiArray?, assets: ModelAssets) throws -> [String: MLFeatureValue] {
        let l = assets.sequenceBuckets.first(where: { $0 >= row.count })!, k = assets.adapter.optionCapacity
        func integers(_ values: [Int], _ shape: [Int]) throws -> MLFeatureValue {
            let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
            for (i, value) in values.enumerated() { array[i] = NSNumber(value: value) }
            return MLFeatureValue(multiArray: array)
        }
        var result = [
            "input_ids": try integers(row.ids + Array(repeating: assets.adapter.padID, count: l - row.count), [1, l]),
            "option_indices": try integers(row.options + Array(repeating: 0, count: k - row.options.count), [1, k]),
        ]
        if assets.tensors == "masked-options" {
            result["token_mask"] = try integers(Array(repeating: 1, count: row.count) + Array(repeating: 0, count: l - row.count), [1, l])
            result["option_mask"] = try integers(Array(repeating: 1, count: row.options.count) + Array(repeating: 0, count: k - row.options.count), [1, k])
            result["question_type"] = try integers([row.type], [1])
        } else {
            result["position_ids"] = try integers(Array(0..<row.count) + Array(repeating: 0, count: l - row.count), [1, l])
            result["decision_indices"] = try integers([row.count - 1], [1])
            let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: l), NSNumber(value: l)], dataType: .float32)
            let values = mask.dataPointer.bindMemory(to: Float.self, capacity: l * l)
            for query in 0..<l {
                for key in 0..<l {
                    values[query * l + key] = (key <= query && key < row.count) || query == key ? 0 : -10000
                }
            }
            result["attention_bias"] = MLFeatureValue(multiArray: mask)
        }
        if let imagePixels { result["image_pixels"] = MLFeatureValue(multiArray: imagePixels) }
        return result
    }
}

/// Lock-protected admission happens before the actor hop, so its mailbox stays bounded.
final class RequestAdmission: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private var pending = 0
    init(limit: Int) { self.limit = limit }
    func acquire() throws {
        lock.lock(); defer { lock.unlock() }
        guard pending < limit else { throw SwevError.queueFull }
        pending += 1
    }
    func release() {
        lock.lock(); defer { lock.unlock() }
        pending -= 1
    }
}

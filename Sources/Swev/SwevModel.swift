import Foundation

/// Identity and decision limits of a loaded model, independent of its storage format.
public struct ModelDescriptor: Sendable {
    public let id: String
    public let revision: String
    public let capabilities: ModelCapabilities
}

/// A resident decision model. Keep it loaded across predictions to avoid repeated initialization.
/// Loads ordinary supported MLX models from the Hub or a local directory.
public final class SwevModel: Sendable {
    public let descriptor: ModelDescriptor
    private let runtime: any DecisionRuntime
    private let admission: RequestAdmission

    private init(runtime: any DecisionRuntime, descriptor: ModelDescriptor, maxPendingRequests: Int) {
        self.runtime = runtime
        self.descriptor = descriptor
        admission = RequestAdmission(limit: maxPendingRequests)
    }

    /// Loads an ordinary supported MLX model from Hugging Face, reusing its normal download cache.
    /// Pin `revision` to a commit for reproducibility. Supported architectures depend on the runtime.
    /// `maxContextTokens` bounds input further, or supplies a missing model context limit.
    /// Requests are serialized and bounded; `maxPendingRequests` must be between 1 and 64.
    public static func load(hf modelID: String, revision: String = "main", maxPendingRequests: Int = 8, maxContextTokens: Int? = nil) async throws -> SwevModel {
        try validateAdmission(maxPendingRequests)
        return try await loaded(MLXDecisionBackend.load(hf: modelID, revision: revision, contextLimit: maxContextTokens), maxPendingRequests: maxPendingRequests)
    }

    /// Loads a local MLX model directory containing weights, configuration, and tokenizer files.
    /// `maxContextTokens` has the same meaning as in `load(hf:)`.
    /// No download or conversion is performed. The URL must be a file URL pointing to a directory.
    public static func load(url: URL, maxPendingRequests: Int = 8, maxContextTokens: Int? = nil) async throws -> SwevModel {
        try validateAdmission(maxPendingRequests)
        return try await loaded(MLXDecisionBackend.load(url: url, contextLimit: maxContextTokens), maxPendingRequests: maxPendingRequests)
    }

    private static func validateAdmission(_ limit: Int) throws {
        guard (1...64).contains(limit) else { throw SwevError.invalidRequest("Pending request limit must be 1–64") }
    }

    private static func loaded(_ runtime: any DecisionRuntime, maxPendingRequests: Int) -> SwevModel {
        .init(runtime: runtime, descriptor: .init(id: runtime.id, revision: runtime.revision,
            capabilities: .init(modalities: runtime.supportsImages ? ["text", "image"] : ["text"],
                questionTypes: QuestionType.allCases, limits: .init(maxQuestionsPerRequest: 64,
                    maxOptionsPerQuestion: 26, maxSequenceTokens: runtime.maxContextTokens))),
            maxPendingRequests: maxPendingRequests)
    }

    /// Evaluates independent questions, preserving their order and application option IDs.
    /// No answer text is generated. Probabilities are normalized only over the supplied candidates.
    public func predict(_ request: DecisionRequest) async throws -> DecisionResponse {
        try Task.checkCancellation()
        try admission.acquire()
        defer { admission.release() }
        guard request.questions.count <= descriptor.capabilities.limits.maxQuestionsPerRequest else { throw SwevError.resourceLimit }
        return try await runtime.predict(request)
    }

    /// Convenience form of `predict(_:)`, including optional image bytes and echoed metadata.
    public func predict(state: JSONValue, questions: [Question], images: [ImageInput] = [],
                        metadata: RequestMetadata? = nil) async throws -> DecisionResponse {
        try await predict(.init(state: state, questions: questions, images: images, metadata: metadata))
    }
}

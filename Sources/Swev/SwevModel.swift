import CoreML
import Foundation

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

public struct RuntimeConfiguration: Sendable {
    public var computeUnits: ComputeUnits

    public init(computeUnits: ComputeUnits = .all) {
        self.computeUnits = computeUnits
    }
}

public struct ModelCapabilities: Decodable, Sendable {
    public let modalities: [String]
    public let questionTypes: [QuestionType]
    public let limits: Limits
    public var supportsImages: Bool { modalities.contains("image") }

    public struct Limits: Decodable, Sendable {
        public let maxQuestionsPerRequest: Int
        public let maxOptionsPerQuestion: Int
        public let maxSequenceTokens: Int
    }
}

public struct ModelDescriptor: Decodable, Sendable {
    public let contractVersion: String
    public let modelVersion: String
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
        let descriptor: ModelDescriptor
        do { descriptor = try JSONDecoder().decode(Self.self, from: Data(json.utf8)) }
        catch { throw SwevError.invalidMetadata }
        guard descriptor.contractVersion == "1.0" else {
            throw SwevError.unsupportedContractVersion(descriptor.contractVersion)
        }
        let limits = descriptor.capabilities.limits
        guard !descriptor.id.isEmpty, !descriptor.modelVersion.isEmpty,
              !descriptor.execution.profile.isEmpty, !descriptor.execution.inputAdapter.isEmpty,
              limits.maxQuestionsPerRequest > 0, limits.maxOptionsPerQuestion >= 2,
              limits.maxSequenceTokens > 0 else { throw SwevError.invalidMetadata }
        return descriptor
    }
}

/// API scaffold. No executable model adapter is registered yet.
public actor SwevModel {
    public nonisolated let descriptor: ModelDescriptor

    private init(descriptor: ModelDescriptor) { self.descriptor = descriptor }

    /// Compiles standard assets and reads embedded configuration. Currently throws
    /// `unsupportedProfile` after inspection: text tokenization and inference are not wired.
    public static func load(from modelURL: URL, configuration: RuntimeConfiguration = .init()) async throws -> SwevModel {
        try Task.checkCancellation()
        guard modelURL.isFileURL,
              ["mlpackage", "mlmodel", "mlmodelc"].contains(modelURL.pathExtension),
              FileManager.default.fileExists(atPath: modelURL.path) else {
            throw SwevError.invalidModelAsset
        }
        let descriptor = try await Task.detached {
            let needsCompilation = modelURL.pathExtension != "mlmodelc"
            let compiledURL = needsCompilation ? try MLModel.compileModel(at: modelURL) : modelURL
            defer {
                if needsCompilation { try? FileManager.default.removeItem(at: compiledURL) }
            }
            let config = MLModelConfiguration()
            config.computeUnits = configuration.computeUnits.coreML
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
            return try ModelDescriptor.read(metadata: metadata)
        }.value
        try Task.checkCancellation()
        throw SwevError.unsupportedProfile(descriptor.execution.profile)
    }

    public func predict(state: JSONValue, questions: [Question], images: [ImageInput] = [],
                        metadata: RequestMetadata? = nil) async throws -> DecisionResponse {
        try await predict(.init(state: state, questions: questions, images: images, metadata: metadata))
    }

    public func predict(_ request: DecisionRequest) async throws -> DecisionResponse {
        try Task.checkCancellation()
        guard request.images.isEmpty else { throw SwevError.unsupportedModality }
        try request.validate()
        throw SwevError.unsupportedProfile(descriptor.execution.profile)
    }
}

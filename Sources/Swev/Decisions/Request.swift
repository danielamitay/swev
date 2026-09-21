import Foundation

/// The three decision primitives supported by Swev's request and response APIs.
public enum QuestionType: String, Codable, Sendable, CaseIterable {
    case choice, score, noul
}

/// A candidate identified by an application-defined ID and optional model-facing description.
public struct ChoiceOption: Sendable {
    public let id: String
    public let description: JSONValue?

    public init(id: String, description: JSONValue? = nil) {
        self.id = id
        self.description = description
    }
}

/// A question evaluated independently against the request's shared state and image.
/// IDs must be nonempty and unique within a request. Candidate capacity comes from the model.
public enum Question: Sendable {
    /// Select among at least two options. Option IDs must be nonempty and unique.
    case choice(id: String, instructions: JSONValue, options: [ChoiceOption])
    /// Rate ordered levels, from lowest to highest. The answer is an expected zero-based index.
    case score(id: String, instructions: JSONValue, levels: [JSONValue])
    /// Estimate the probability of true; optional descriptions define the false/true alternatives.
    case noul(id: String, instructions: JSONValue, falseDescription: JSONValue? = nil, trueDescription: JSONValue? = nil)

    public var id: String {
        switch self {
        case .choice(let id, _, _), .score(let id, _, _), .noul(let id, _, _, _): return id
        }
    }

    public var type: QuestionType {
        switch self {
        case .choice: return .choice
        case .score: return .score
        case .noul: return .noul
        }
    }

    public var instructions: JSONValue {
        switch self {
        case .choice(_, let value, _), .score(_, let value, _), .noul(_, let value, _, _): return value
        }
    }

    public var optionCount: Int {
        switch self {
        case .choice(_, _, let options): return options.count
        case .score(_, _, let levels): return levels.count
        case .noul: return 2
        }
    }
}

/// Owned encoded image bytes. Support and preprocessing are defined by the loaded model runtime.
/// PNG/JPEG inputs are limited to 32 MiB, 8,192 pixels per side, and 16,777,216 total pixels.
public struct ImageInput: Sendable {
    public let data: Data
    /// The encoded format: `image/png` or `image/jpeg`. It must match the bytes.
    public let contentType: String

    public init(data: Data, contentType: String) {
        self.data = data
        self.contentType = contentType
    }
}

/// Caller-owned identifiers echoed in the typed response; never included in the model prompt.
public struct RequestMetadata: Sendable {
    public let sourceID: String?
    public let schemaRevision: String?

    public init(sourceID: String? = nil, schemaRevision: String? = nil) {
        self.sourceID = sourceID
        self.schemaRevision = schemaRevision
    }
}

/// Shared context and ordered questions for one prediction request.
/// State and instructions may be strings, objects, or arrays. At most one image is supported.
public struct DecisionRequest: Sendable {
    public let state: JSONValue
    public let questions: [Question]
    public let images: [ImageInput]
    public let metadata: RequestMetadata?

    public init(state: JSONValue, questions: [Question], images: [ImageInput] = [], metadata: RequestMetadata? = nil) {
        self.state = state
        self.questions = questions
        self.images = images
        self.metadata = metadata
    }

    /// Checks question IDs, candidate counts, and JSON values without loading a model.
    /// Model-specific token, option, image, and request limits are checked during prediction.
    /// - Throws: `SwevError.invalidRequest` for invalid structure or values.
    public func validate() throws {
        try state.validateContext()
        guard !questions.isEmpty else { throw SwevError.invalidRequest("At least one question is required") }
        guard Set(questions.map(\.id)).count == questions.count else {
            throw SwevError.invalidRequest("Duplicate question IDs")
        }
        for question in questions {
            guard !question.id.isEmpty else { throw SwevError.invalidRequest("Empty question ID") }
            try question.instructions.validateContext()
            guard question.optionCount >= 2 else { throw SwevError.invalidRequest("At least two candidates are required") }
            switch question {
            case .choice(_, _, let options):
                guard options.allSatisfy({ !$0.id.isEmpty }), Set(options.map(\.id)).count == options.count else {
                    throw SwevError.invalidRequest("Empty or duplicate option IDs")
                }
                for option in options { _ = try option.description?.jsonString() }
            case .score(_, _, let levels):
                for level in levels { _ = try level.jsonString() }
            case .noul(_, _, let falseDescription, let trueDescription):
                _ = try falseDescription?.jsonString()
                _ = try trueDescription?.jsonString()
            }
        }
    }
}

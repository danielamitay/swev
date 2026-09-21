import Foundation

/// An option's ID and normalized probability, in the original candidate order.
public struct CandidateProbability: Sendable {
    public let id: String
    public let probability: Double
}

/// The model-declared statistic used to summarize a probability distribution.
public enum ConfidenceMethod: String, Sendable, Codable {
    case normalizedEntropy = "normalized_entropy_v1"
    case chanceAdjustedMax = "chance_adjusted_max_v1"
    case modalDistance = "modal_distance_v1"
}

/// A distribution statistic in 0...1, not a calibrated probability of correctness.
public struct Confidence: Sendable {
    public let value: Double
    public let method: ConfidenceMethod
}

/// The most probable option, its probability, and the complete candidate distribution.
public struct ChoiceAnswer: Sendable {
    public let choice: String
    public let probabilities: [CandidateProbability]
    public let selectedProbability: Double
    public let confidence: Confidence
}

/// A binary decision expressed as a probability, leaving threshold selection to the caller.
public struct NoulAnswer: Sendable {
    /// Probability of true, not a thresholded Boolean.
    public let noul: Double
}

/// An ordinal decision. Probabilities and legend entries follow the supplied level order.
public struct ScoreAnswer: Sendable {
    /// Expected zero-based ordinal level.
    public let score: Double
    public let modalLevel: Int
    public let probabilities: [Double]
    public let legend: [String]
    public let confidence: Confidence
}

/// A typed answer associated with its request question ID.
public enum Answer: Sendable {
    case choice(id: String, ChoiceAnswer)
    case score(id: String, ScoreAnswer)
    case noul(id: String, NoulAnswer)

    public var id: String {
        switch self {
        case .choice(let id, _), .score(let id, _), .noul(let id, _): return id
        }
    }
}

/// Logical token counts across all questions, excluding padding.
public struct TokenUsage: Sendable {
    /// Sum of encoded prompt lengths; shared state is counted again for each question.
    public let inputTokens: Int
    /// Zero for decision models, which score candidates without generating text.
    public let outputTokens: Int
}

/// Answers in request order, with model identity, token usage, and echoed caller metadata.
public struct DecisionResponse: Sendable {
    public let modelID: String
    /// The checkpoint revision, falling back to the export's model version when absent.
    public let modelRevision: String
    public let answers: [Answer]
    public let usage: TokenUsage
    public let metadata: RequestMetadata?

    /// Returns a choice answer by question ID.
    /// - Throws: `SwevError.missingAnswer` or `SwevError.answerTypeMismatch`.
    public func choice(_ id: String) throws -> ChoiceAnswer {
        guard case .choice(_, let value) = try answer(id) else {
            throw SwevError.answerTypeMismatch(id: id, expected: .choice)
        }
        return value
    }

    /// Returns an ordinal score by question ID.
    /// - Throws: `SwevError.missingAnswer` or `SwevError.answerTypeMismatch`.
    public func score(_ id: String) throws -> ScoreAnswer {
        guard case .score(_, let value) = try answer(id) else {
            throw SwevError.answerTypeMismatch(id: id, expected: .score)
        }
        return value
    }

    /// Returns the probability of true for a question ID.
    /// - Throws: `SwevError.missingAnswer` or `SwevError.answerTypeMismatch`.
    public func noul(_ id: String) throws -> NoulAnswer {
        guard case .noul(_, let value) = try answer(id) else {
            throw SwevError.answerTypeMismatch(id: id, expected: .noul)
        }
        return value
    }

    private func answer(_ id: String) throws -> Answer {
        guard let answer = answers.first(where: { $0.id == id }) else { throw SwevError.missingAnswer(id) }
        return answer
    }
}

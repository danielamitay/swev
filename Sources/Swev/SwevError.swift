/// Runtime validation and execution failures. MLX, filesystem, tokenizer, and cancellation errors may also propagate.
public enum SwevError: Error, Sendable, Equatable {
    case unsupportedTokenizer
    case queueFull
    /// A bounded input, image, or request-count limit was exceeded.
    case resourceLimit
    /// The rendered prompt or one of its fields exceeds the model's token budget.
    case contextOverflow
    case tooManyOptions(limit: Int)
    case inferenceFailed
    case invalidModelAsset
    case unsupportedModality
    case invalidRequest(String)
    case missingAnswer(String)
    case answerTypeMismatch(id: String, expected: QuestionType)
    case nonFiniteOutput
}

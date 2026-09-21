/// Runtime validation and execution failures. Core ML, filesystem, and cancellation errors may also propagate.
public enum SwevError: Error, Sendable, Equatable {
    case unsupportedTokenizer
    case queueFull
    /// A bounded input, image, or request-count limit was exceeded.
    case resourceLimit
    /// The rendered prompt or one of its fields exceeds the active route's token budget.
    case contextOverflow
    case tooManyOptions(limit: Int)
    case signatureMismatch
    case metadataIntegrityFailure
    case inferenceFailed
    case invalidModelAsset
    case missingMetadata(key: String)
    case invalidMetadata
    case unsupportedContractVersion(String)
    case unsupportedProfile(String)
    case unsupportedModality
    case invalidRequest(String)
    case missingAnswer(String)
    case answerTypeMismatch(id: String, expected: QuestionType)
    case nonFiniteOutput
}

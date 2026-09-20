public enum SwevError: Error, Sendable, Equatable {
    case unsupportedTokenizer
    case queueFull
    case resourceLimit
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

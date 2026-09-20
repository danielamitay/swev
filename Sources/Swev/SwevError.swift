public enum SwevError: Error, Sendable, Equatable {
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

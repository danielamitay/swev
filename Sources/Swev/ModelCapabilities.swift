/// Modalities, question types, and limits reported by the loaded model.
public struct ModelCapabilities: Decodable, Sendable {
    public let modalities: [String]
    public let questionTypes: [QuestionType]
    public let limits: Limits
    public var supportsImages: Bool { modalities.contains("image") }

    public struct Limits: Decodable, Sendable {
        public let maxQuestionsPerRequest: Int
        public let maxOptionsPerQuestion: Int
        /// Maximum tokens per question, including formatting and image tokens. Field limits may be lower.
        public let maxSequenceTokens: Int
    }
}


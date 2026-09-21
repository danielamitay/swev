import Foundation

/// A deterministic, model-independent question and its candidate-token mapping.
/// Tokenization and chat framing belong to the runtime; application IDs never become labels.
struct DecisionPrompt: Sendable {
    let text: String
    let labels: [String]

    init(state: JSONValue, question: Question) throws {
        guard (2...26).contains(question.optionCount) else {
            throw SwevError.tooManyOptions(limit: 26)
        }
        labels = (0..<question.optionCount).map { String(UnicodeScalar(65 + $0)!) }
        let descriptions: [JSONValue]
        switch question {
        case .choice(_, _, let options):
            descriptions = options.map { $0.description ?? .string($0.id) }
        case .score(_, _, let levels):
            descriptions = levels
        case .noul(_, _, let falseDescription, let trueDescription):
            descriptions = [falseDescription ?? .string("false"), trueDescription ?? .string("true")]
        }
        let options = try zip(labels, descriptions).map { label, description in
            "\(label): \(try description.jsonString())"
        }.joined(separator: "\n")
        text = """
        Make one decision using the supplied state and any attached images.
        Select the best answer to the question from the options below.
        Reply with only its uppercase letter, without explanation or punctuation.

        State:
        \(try state.jsonString())

        Question:
        \(try question.instructions.jsonString())

        Options (in order):
        \(options)
        """
    }

    /// Reject unsupported tokenizers instead of silently comparing partial or duplicate labels.
    func candidateTokenIDs(encode: (String) throws -> [Int]) throws -> [Int] {
        let encoded = try labels.map(encode)
        guard encoded.allSatisfy({ $0.count == 1 && $0[0] >= 0 }) else {
            throw SwevError.unsupportedTokenizer
        }
        let ids = encoded.map { $0[0] }
        guard Set(ids).count == ids.count else { throw SwevError.unsupportedTokenizer }
        return ids
    }
}

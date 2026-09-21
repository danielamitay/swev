import Foundation

/// A deterministic, model-independent question and its candidate-token mapping.
/// Tokenization and chat framing belong to the runtime; application IDs never become labels.
struct DecisionPrompt: Sendable {
    static let answerPrefix = " Answer: "
    let text: String
    let labels: [String]

    init(state: JSONValue, question: Question) throws {
        guard (2...26).contains(question.optionCount) else {
            throw SwevError.tooManyOptions(limit: 26)
        }
        labels = (0..<question.optionCount).map { String(UnicodeScalar(65 + $0)!) }
        func text(_ value: JSONValue) throws -> String {
            if case .string(let string) = value { return string }
            return try value.jsonString()
        }
        func description(_ value: JSONValue?) throws -> String? {
            guard let value, !value.isNull, !value.isEmptyString else { return nil }
            return try text(value)
        }
        let options: [String]
        switch question {
        case .choice(_, _, let candidates):
            options = try candidates.map { option in
                try option.id + (description(option.description).map { ": " + $0 } ?? "")
            }
        case .score(_, _, let levels): options = try levels.map(text)
        case .noul(_, _, let no, let yes):
            options = [try description(no) ?? "No", try description(yes) ?? "Yes"]
        }
        self.text = "Choose the best answer to the question using the state below. Reply with only the answer letter.\n\nState: "
            + (try text(state)) + "\n\nQuestion: " + (try text(question.instructions)) + "\n\nAnswers:\n"
            + zip(labels, options).map { "\($0). \($1)\n" }.joined() + "\nAnswer:"

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

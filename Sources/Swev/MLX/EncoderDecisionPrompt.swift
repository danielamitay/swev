import Foundation

/// Marker-based decision prompt. Exceeding a source-runtime truncation boundary is an error.
struct EncoderDecisionPrompt {
    let tokens: [Int]
    let markers: [Int]
    let typeIndex: Int

    init(state: JSONValue, question: Question, cls: Int, sep: Int, marker: Int, maskText: String,
         headLimit: Int, contextLimit: Int, encode: (String) -> [Int]) throws {
        func text(_ value: JSONValue) throws -> String {
            if case .string(let string) = value { return string }
            return try value.spacedJSON()
        }
        func ids(_ string: String) -> [Int] { encode(string.replacingOccurrences(of: maskText, with: " ")) }
        let instructions: JSONValue
        let options: [String]
        switch question {
        case .choice(_, let prompt, let choices):
            typeIndex = 0; instructions = prompt
            options = try choices.map { choice in
                guard let description = choice.description, !description.isNull, !description.isEmptyString else { return choice.id }
                return try choice.id + ": " + text(description)
            }
        case .score(_, let prompt, let levels):
            typeIndex = 1; instructions = prompt
            options = try levels.enumerated().map { try "level \($0.offset): " + text($0.element) }
        case .noul(_, let prompt, let no, let yes):
            typeIndex = 2; instructions = prompt
            func description(_ value: JSONValue?, fallback: String) throws -> String {
                guard let value, !value.isNull, !value.isEmptyString else { return fallback }
                return try text(value)
            }
            options = try ["false: " + description(no, fallback: "no, the statement does not hold"),
                           "true: " + description(yes, fallback: "yes, the statement holds")]
        }
        // The source format uses ASCII-escaped JSON for structured instructions only.
        let instructionText: String
        if case .string(let string) = instructions { instructionText = string }
        else {
            instructionText = try instructions.spacedJSON().unicodeScalars.map { scalar in
                if scalar.value < 128 { return String(scalar) }
                return String(scalar).utf16.map { String(format: "\\u%04x", $0) }.joined()
            }.joined()
        }
        let head = ids(question.type.rawValue + " question: " + instructionText)
        let optionTokens = options.map { [marker] + ids(" " + $0) }
        let remaining = headLimit - optionTokens.reduce(0) { $0 + $1.count }
        guard remaining >= 16, head.count <= max(8, remaining), optionTokens.allSatisfy({ $0.count <= 49 }) else {
            throw SwevError.contextOverflow
        }
        var result = [cls] + head + [sep], positions: [Int] = []
        for option in optionTokens { positions.append(result.count); result += option }
        result += [sep] + ids(try text(state)) + [sep]
        guard result.count <= contextLimit else { throw SwevError.contextOverflow }
        tokens = result; markers = positions
    }
}

import Foundation

/// Ordered text-only request/response subset. No remote URL fetching or model aliases.
public enum DecisionCodec {
    public static func decodeRequest(_ data: Data, modelID: String? = nil) throws -> DecisionRequest {
        let root = try JSONValue.parse(data)
        if let supplied = root.member("model") {
            guard case .string(let id) = supplied, id == modelID else { throw SwevError.invalidRequest("Model ID mismatch") }
        }
        guard root.member("images") == nil else { throw SwevError.unsupportedModality }
        guard let state = root.member("state"), case .object(let questions) = root.member("questions") else {
            throw SwevError.invalidRequest("Missing state or questions")
        }
        let parsed = try questions.map { id, value -> Question in
            guard case .string(let type) = value.member("type"), let instructions = value.member("instructions") else {
                throw SwevError.invalidRequest("Missing question type or instructions")
            }
            switch type {
            case "choice":
                guard case .object(let options) = value.member("criteria") else { throw SwevError.invalidRequest("Choice criteria must be an object") }
                return .choice(id: id, instructions: instructions, options: options.map { .init(id: $0.0, description: $0.1) })
            case "score":
                guard case .array(let levels) = value.member("criteria"), levels.count <= 10 else { throw SwevError.invalidRequest("Score requires 2–10 levels") }
                return .score(id: id, instructions: instructions, levels: levels)
            case "noul":
                let criteria = value.member("criteria")
                if let criteria, !criteria.isNull {
                    guard case .object = criteria else { throw SwevError.invalidRequest("Noul criteria must be an object") }
                }
                return .noul(id: id, instructions: instructions, falseDescription: criteria?.member("false"), trueDescription: criteria?.member("true"))
            default: throw SwevError.invalidRequest("Unknown question type")
            }
        }
        let request = DecisionRequest(state: state, questions: parsed)
        try request.validate()
        return request
    }

    public static func encodeResponse(_ response: DecisionResponse) throws -> Data {
        let answers = response.answers.map { answer -> (String, JSONValue) in
            let fields: [(String, JSONValue)]
            switch answer {
            case .choice(_, let value):
                fields = [("type", "choice"), ("choice", .string(value.choice)),
                          ("probabilities", .object(value.probabilities.map { ($0.id, .number($0.probability)) })),
                          ("confidence", .number(value.confidence.value))]
            case .noul(_, let value): fields = [("type", "noul"), ("noul", .number(value.noul))]
            case .score(_, let value):
                fields = [("type", "score"), ("score", .number(value.score)),
                          ("probabilities", .object(value.probabilities.enumerated().map { (String($0.offset), .number($0.element)) })),
                          ("legend", .object(value.legend.enumerated().map { (String($0.offset), .string($0.element)) })),
                          ("confidence", .number(value.confidence.value))]
            }
            return (answer.id, .object(fields))
        }
        return Data(try JSONValue.object([
            ("model", .string(response.modelID)), ("answers", .object(answers)),
            ("usage", .object([("input_tokens", .number(Double(response.usage.inputTokens))), ("output_tokens", .number(Double(response.usage.outputTokens)))])),
        ]).jsonString().utf8)
    }
}

extension JSONValue {
    func member(_ key: String) -> JSONValue? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.0 == key }?.1
    }

    /// Parses JSON without losing object order; rejects duplicate keys and nonfinite numbers.
    public static func parse(_ data: Data) throws -> JSONValue {
        guard data.count <= 1_048_576 else { throw SwevError.resourceLimit }
        var parser = OrderedJSONParser(bytes: Array(data))
        let value = try parser.value(depth: 0)
        parser.whitespace()
        guard parser.offset == parser.bytes.count else { throw SwevError.invalidRequest("Trailing JSON data") }
        return value
    }
}

private struct OrderedJSONParser {
    let bytes: [UInt8]
    var offset = 0
    var current: UInt8? { offset < bytes.count ? bytes[offset] : nil }
    mutating func whitespace() { while let current, [9, 10, 13, 32].contains(current) { offset += 1 } }
    mutating func expect(_ byte: UInt8) throws {
        guard current == byte else { throw SwevError.invalidRequest("Malformed JSON") }
        offset += 1
    }
    mutating func string() throws -> String {
        let start = offset
        try expect(34)
        while let current {
            offset += 1
            if current == 92 { guard offset < bytes.count else { break }; offset += 1 }
            else if current == 34 {
                do { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<offset])) }
                catch { throw SwevError.invalidRequest("Malformed JSON string") }
            }
        }
        throw SwevError.invalidRequest("Unterminated JSON string")
    }
    mutating func value(depth: Int) throws -> JSONValue {
        guard depth <= 64 else { throw SwevError.resourceLimit }
        whitespace()
        switch current {
        case 34: return .string(try string())
        case 123:
            offset += 1; whitespace()
            var fields: [(String, JSONValue)] = []; var keys: Set<String> = []
            if current == 125 { offset += 1; return .object(fields) }
            while true {
                whitespace(); let key = try string()
                guard keys.insert(key).inserted else { throw SwevError.invalidRequest("Duplicate JSON object keys") }
                whitespace(); try expect(58)
                fields.append((key, try value(depth: depth + 1))); whitespace()
                if current == 125 { offset += 1; break }; try expect(44)
            }
            return .object(fields)
        case 91:
            offset += 1; whitespace(); var values: [JSONValue] = []
            if current == 93 { offset += 1; return .array(values) }
            while true {
                values.append(try value(depth: depth + 1)); whitespace()
                if current == 93 { offset += 1; break }; try expect(44)
            }
            return .array(values)
        default:
            let start = offset
            while let current, ![9, 10, 13, 32, 44, 93, 125].contains(current) { offset += 1 }
            let literal = Data(bytes[start..<offset])
            if literal == Data("null".utf8) { return .null }
            if literal == Data("true".utf8) { return .bool(true) }
            if literal == Data("false".utf8) { return .bool(false) }
            if let number = try? JSONDecoder().decode(Double.self, from: literal), number.isFinite { return .number(number) }
            throw SwevError.invalidRequest("Malformed JSON value")
        }
    }
}

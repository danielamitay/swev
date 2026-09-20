import Foundation

func pythonString(_ value: JSONValue) throws -> String {
    switch value {
    case .string(let text): return text
    case .bool(let value): return value ? "True" : "False"
    case .null: return "None"
    case .number: return try value.jsonString()
    case .array(let items): return "[" + (try items.map { try pythonRepr($0) }.joined(separator: ", ")) + "]"
    case .object(let fields): return "{" + (try fields.map { try pythonRepr(.string($0.0)) + ": " + pythonRepr($0.1) }.joined(separator: ", ")) + "}"
    }
}

private func pythonRepr(_ value: JSONValue) throws -> String {
    guard case .string(let text) = value else { return try pythonString(value) }
    let quote = text.contains("'") && !text.contains("\"") ? "\"" : "'"
    var result = quote
    for scalar in text.unicodeScalars {
        switch scalar.value {
        case 9: result += "\\t"
        case 10: result += "\\n"
        case 13: result += "\\r"
        case 92: result += "\\\\"
        default:
            if String(scalar) == quote { result += "\\" + quote }
            else if [.control, .format, .privateUse, .unassigned, .surrogate, .lineSeparator, .paragraphSeparator, .spaceSeparator].contains(scalar.properties.generalCategory) && scalar.value != 32 {
                if scalar.value <= 255 { result += String(format: "\\x%02x", scalar.value) }
                else if scalar.value <= 65535 { result += String(format: "\\u%04x", scalar.value) }
                else { result += String(format: "\\U%08x", scalar.value) }
            } else { result += String(scalar) }
        }
    }
    return result + quote
}

extension JSONValue {
    var isNull: Bool { if case .null = self { return true }; return false }
    var isEmptyString: Bool { if case .string("") = self { return true }; return false }
    var isTruthy: Bool {
        switch self {
        case .null: return false
        case .bool(let value): return value
        case .number(let value): return value != 0
        case .string(let value): return !value.isEmpty
        case .array(let value): return !value.isEmpty
        case .object(let value): return !value.isEmpty
        }
    }
    func spacedJSON() throws -> String {
        switch self {
        case .array(let items): return "[" + (try items.map { try $0.spacedJSON() }.joined(separator: ", ")) + "]"
        case .object(let fields): return "{" + (try fields.map { try JSONValue.string($0.0).jsonString() + ": " + $0.1.spacedJSON() }.joined(separator: ", ")) + "}"
        default: return try jsonString()
        }
    }
    func indentedText(indent: Int = 0) throws -> String {
        let pad = String(repeating: "  ", count: indent)
        switch self {
        case .null: return ""
        case .array(let items): return try items.map { try pad + "- " + $0.indentedText(indent: indent + 1).drop(while: { $0.isWhitespace }) }.joined(separator: "\n")
        case .object(let fields): return try fields.map { key, value in
            switch value {
            case .array, .object: return try pad + key + ":\n" + value.indentedText(indent: indent + 1)
            default: return try pad + key + ": " + value.indentedText()
            }
        }.joined(separator: "\n")
        default: return try pythonString(self)
        }
    }
}

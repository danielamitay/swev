import Foundation

/// JSON with ordered object members. Duplicate keys and nonfinite numbers are invalid.
public indirect enum JSONValue: Sendable, ExpressibleByStringLiteral {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([(String, JSONValue)])

    public init(stringLiteral value: String) { self = .string(value) }

    /// Deterministic JSON rendering, preserving caller-supplied object order.
    public func jsonString() throws -> String {
        switch self {
        case .string(let value):
            return String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
        case .number(let value):
            guard value.isFinite else { throw SwevError.invalidRequest("Nonfinite JSON number") }
            return String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        case .array(let values): return "[" + (try values.map { try $0.jsonString() }.joined(separator: ",")) + "]"
        case .object(let members):
            guard Set(members.map(\.0)).count == members.count else {
                throw SwevError.invalidRequest("Duplicate JSON object keys")
            }
            return "{" + (try members.map {
                try JSONValue.string($0.0).jsonString() + ":" + $0.1.jsonString()
            }.joined(separator: ",")) + "}"
        }
    }

    func validateContext() throws {
        switch self {
        case .string, .object, .array: _ = try jsonString()
        default: throw SwevError.invalidRequest("State and instructions must be strings, objects, or arrays")
        }
    }
}

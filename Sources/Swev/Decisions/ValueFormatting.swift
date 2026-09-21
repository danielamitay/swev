extension JSONValue {
    var isNull: Bool { if case .null = self { return true }; return false }
    var isEmptyString: Bool { if case .string("") = self { return true }; return false }
    func spacedJSON() throws -> String {
        switch self {
        case .array(let items): return "[" + (try items.map { try $0.spacedJSON() }.joined(separator: ", ")) + "]"
        case .object(let fields): return "{" + (try fields.map { try JSONValue.string($0.0).jsonString() + ": " + $0.1.spacedJSON() }.joined(separator: ", ")) + "}"
        default: return try jsonString()
        }
    }
}

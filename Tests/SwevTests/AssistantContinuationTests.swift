import Testing
@testable import Swev

@Test func decisionContentCompletesChannelHeadersWithoutGenerating() throws {
    let controls = ["<|start|>": 1000, "<|channel|>": 1001, "<|message|>": 1002]
    func encode(_ text: String) -> [Int] {
        var remaining = text[...]
        var result: [Int] = []
        while !remaining.isEmpty {
            if let (token, id) = controls.first(where: { remaining.hasPrefix($0.key) }) {
                result.append(id)
                remaining = remaining.dropFirst(token.count)
            } else {
                result.append(Int(remaining.removeFirst().asciiValue!))
            }
        }
        return result
    }
    func decode(_ ids: [Int]) -> String {
        ids.map { id in controls.first(where: { $0.value == id })?.key ?? String(UnicodeScalar(id)!) }.joined()
    }
    let continuation = try AssistantContinuation(encode: encode, decode: decode)
    let open = encode("<|start|>assistant")
    #expect(try decode(continuation.tokens(after: open)) == "<|channel|>final<|message|> Answer: ")
    let complete = encode("<|start|>assistant<|channel|>final<|message|>")
    #expect(try decode(continuation.tokens(after: complete)) == " Answer: ")
    #expect(throws: SwevError.self) {
        try continuation.tokens(after: encode("<|start|>assistant<|channel|>analysis<|message|>"))
    }
    let plain = try AssistantContinuation(encode: { $0.utf8.map(Int.init) }, decode: { String(bytes: $0.map(UInt8.init), encoding: .utf8)! })
    #expect(plain.tailCount == 0)
    #expect(try plain.tokens(after: []) == Array(" Answer: ".utf8).map(Int.init))
}

@Test func recipientHeaderComesFromTheTemplate() throws {
    let controls = ["<|start|>": 1000, "<|message|>": 1001]
    func encode(_ text: String) -> [Int] {
        var remaining = text[...]
        var result: [Int] = []
        while !remaining.isEmpty {
            if let (token, id) = controls.first(where: { remaining.hasPrefix($0.key) }) {
                result.append(id)
                remaining = remaining.dropFirst(token.count)
            } else { result.append(Int(remaining.removeFirst().asciiValue!)) }
        }
        return result
    }
    func decode(_ ids: [Int]) -> String {
        ids.map { id in controls.first(where: { $0.value == id })?.key ?? String(UnicodeScalar(id)!) }.joined()
    }
    let continuation = try AssistantContinuation(encode: encode, decode: decode, renderAssistant: {
        "<|start|>user<|message|>Ready?<|start|>assistant to=user<|message|>" + AssistantContinuation.probe
    })
    #expect(try decode(continuation.tokens(after: encode("<|start|>assistant"))) == " to=user<|message|> Answer: ")
    #expect(try decode(continuation.tokens(after: encode("<|start|>assistant to=user<|message|>"))) == " Answer: ")
    #expect(throws: SwevError.self) {
        try continuation.tokens(after: encode("<|start|>assistant to=tool<|message|>"))
    }
    #expect(throws: SwevError.self) {
        try AssistantContinuation(encode: encode, decode: decode, renderAssistant: { "missing header" })
    }
}

import Foundation
import Testing
@testable import Swev

@Test func encoderMarkersPreserveOrderAndEscapeLiteralMarkers() throws {
    let prompt = try EncoderDecisionPrompt(state: "[MASK]", question: .choice(id: "pick", instructions: "Choose", options: [.init(id: "first"), .init(id: "second")]),
        cls: 1000, sep: 1001, marker: 1002, maskText: "[MASK]", headLimit: 192, contextLimit: 512,
        encode: { $0.utf8.map(Int.init) })
    #expect(prompt.typeIndex == 0)
    #expect(prompt.markers.count == 2)
    #expect(prompt.markers == prompt.tokens.indices.filter { prompt.tokens[$0] == 1002 })
    #expect(prompt.tokens.first == 1000 && prompt.tokens.last == 1001)
}

@Test func encoderRejectsTruncationInsteadOfDroppingStateOrOptions() {
    for state in [String(repeating: "a", count: 600)] {
        #expect(throws: SwevError.contextOverflow) {
            try EncoderDecisionPrompt(state: .string(state), question: .noul(id: "yes", instructions: "Valid?"),
                cls: 1000, sep: 1001, marker: 1002, maskText: "[MASK]", headLimit: 192, contextLimit: 512,
                encode: { $0.utf8.map(Int.init) })
        }
    }
    #expect(throws: SwevError.contextOverflow) {
        try EncoderDecisionPrompt(state: "short", question: .choice(id: "pick", instructions: "Choose", options: [.init(id: String(repeating: "a", count: 60)), .init(id: "b")]),
            cls: 1000, sep: 1001, marker: 1002, maskText: "[MASK]", headLimit: 192, contextLimit: 512,
            encode: { $0.utf8.map(Int.init) })
    }
}

@Test func encoderFormatVersionsAreCheckedBeforeWeights() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data(#"{"format":"laya-mlx","format_version":2}"#.utf8).write(to: directory.appendingPathComponent("mlx_config.json"))
    #expect(throws: SwevError.invalidRequest("Unsupported decision-encoder format or version")) {
        try EncoderDecisionBackend.configuration(directory: directory, limit: nil)
    }
}

@Test func encoderPreservesSourceUnicodeFormatting() throws {
    var encoded: [String] = []
    _ = try EncoderDecisionPrompt(state: "café", question: .choice(id: "pick",
        instructions: .object([("label", "é🍎")]), options: [.init(id: "oui"), .init(id: "non")]),
        cls: 1000, sep: 1001, marker: 1002, maskText: "[MASK]", headLimit: 192, contextLimit: 512,
        encode: { encoded.append($0); return $0.utf8.map(Int.init) })
    #expect(encoded.first == #"choice question: {"label": "\u00e9\ud83c\udf4e"}"#)
    #expect(encoded.last == "café")
}

import Foundation
import CoreML
import Testing
@testable import Swev

@Test func orderedCodec() throws {
    let text = #"{"state":{"z":[true,null,"café"],"a":4},"questions":{"last":{"type":"choice","instructions":"Choose","criteria":{"z":null,"a":"First"}},"first":{"type":"noul","instructions":"Edible?"}}}"#
    let request = try DecisionCodec.decodeRequest(Data(text.utf8))
    #expect(request.questions.map(\.id) == ["last", "first"])
    guard case .choice(_, _, let options) = request.questions[0] else { Issue.record("Not choice"); return }
    #expect(options.map(\.id) == ["z", "a"])
    #expect(try request.state.jsonString() == #"{"z":[true,null,"café"],"a":4}"#)
    for invalid in [#"{"x":1,"x":2}"#, #"{"x":[1,]}"#, #"{"x":"\uD800"}"#, "[1]garbage", "01", "1e999", "[NaN]"] {
        #expect(throws: SwevError.self) { try JSONValue.parse(Data(invalid.utf8)) }
    }
    #expect(throws: SwevError.self) { try DecisionCodec.decodeRequest(Data(#"{"model":"other","state":"a","questions":{}}"#.utf8), modelID: "test") }
}

@Test func sourceRendering() throws {
    let value: JSONValue = .object([("name", "café"), ("items", .array([.bool(true), .null, .number(3)]))])
    #expect(try value.spacedJSON() == #"{"name": "café", "items": [true, null, 3]}"#)
    #expect(try value.indentedText() == "name: café\nitems:\n  - True\n  - \n  - 3")
    #expect(try pythonString(.array([.string("\u{F0000}")])) == "['\\U000f0000']")
    #expect(try pythonString(value) == "{'name': 'café', 'items': [True, None, 3]}")
    #expect(try JSONValue.string("https://example.com").jsonString() == #""https://example.com""#)
}

private struct Manifest: Decodable {
    let tokenizers: [Tokenizer]
    let adapters: [Adapter]
    struct Tokenizer: Decodable { let path: String; let reference: String }
    struct Adapter: Decodable { let tokenizer: String; let recipe: String; let reference: String }
    let models: [Model]
    struct Model: Decodable { let path: String; let reference: String }
}
private func manifest() throws -> Manifest {
    try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: ProcessInfo.processInfo.environment["SWEV_TEST_MANIFEST"]!)))
}
private let hasModels = ProcessInfo.processInfo.environment["SWEV_TEST_MANIFEST"] != nil

@Test(.enabled(if: hasModels)) func tokenizerReferenceParity() throws {
    struct Fixture: Decodable { let text: String; let ids: [Int] }
    for entry in try manifest().tokenizers {
        let tokenizer = try ByteBPETokenizer(data: Data(contentsOf: URL(fileURLWithPath: entry.path)))
        let url = URL(fileURLWithPath: entry.reference)
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
        for fixture in fixtures {
            #expect(try tokenizer.encode(fixture.text) == fixture.ids, "\(entry.path): \(fixture.text.debugDescription)")
        }
        #expect(throws: SwevError.resourceLimit) { try tokenizer.encode(String(repeating: "x", count: 32769)) }
    }
}

@Test(.enabled(if: hasModels)) func textAdapterReferenceParity() throws {
    let manifest = try manifest()
    for entry in manifest.adapters {
        let tokenizer = try ByteBPETokenizer(data: Data(contentsOf: URL(fileURLWithPath: entry.tokenizer)))
        let recipe = try JSONDecoder().decode(TextRecipe.self, from: Data(contentsOf: URL(fileURLWithPath: entry.recipe)))
        let adapter = try TextAdapter(tokenizer: tokenizer, length: 128, optionCapacity: 4, recipe: recipe)
        let url = URL(fileURLWithPath: entry.reference)
        guard case .array(let fixtures) = try JSONValue.parse(Data(contentsOf: url)) else { Issue.record("Bad fixtures"); return }
        for fixture in fixtures {
            let request = try DecisionCodec.decodeRequest(Data(try fixture.member("request")!.jsonString().utf8))
            let row = try adapter.encode(state: request.state, question: request.questions[0])
            func ints(_ key: String) -> [Int] {
                guard case .array(let values) = fixture.member(key) else { return [] }
                return values.map { if case .number(let x) = $0 { return Int(x) }; return -1 }
            }
            let context = try request.state.jsonString()
            #expect(row.ids == ints("ids"), "\(entry.recipe): \(context)")
            #expect(row.options == ints("options"))
        }
        let question = Question.noul(id: "ignored", instructions: "Is this edible?")
        #expect(throws: SwevError.contextOverflow) { try adapter.encode(state: .string(String(repeating: "apple ", count: 200)), question: question) }
    }
}

@Test(.enabled(if: hasModels)) func endToEndModels() async throws {
    let fixtureURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("fixtures/text-cases.json")
    let cases = try JSONValue.parse(Data(contentsOf: fixtureURL))
    guard case .array(let cases) = cases else { Issue.record("Invalid fixture"); return }
    for entry in try manifest().models {
        let model = try await SwevModel.load(from: URL(fileURLWithPath: entry.path), configuration: .init(computeUnits: .cpuOnly))
        let references = try JSONValue.parse(Data(contentsOf: URL(fileURLWithPath: entry.reference)))
        let rows: [JSONValue]
        if case .array(let items) = references { rows = items }
        else if case .array(let items) = references.member("rows") { rows = items }
        else { Issue.record("Invalid references"); return }
        for (i, fixture) in cases.enumerated() {
            let request = try DecisionCodec.decodeRequest(Data(try fixture.member("request")!.jsonString().utf8))
            let response = try await model.predict(request)
            let expected: [Double]
            if case .array(let values) = rows[i].member("probabilities") {
                expected = values.map { if case .number(let x) = $0 { return x }; return .nan }
            } else { Issue.record("Missing reference"); return }
            let actual: [Double]
            switch response.answers[0] {
            case .noul(_, let x): actual = [1 - x.noul, x.noul]
            case .choice(_, let x): actual = x.probabilities.map(\.probability)
            case .score(_, let x): actual = x.probabilities
            }
            #expect(actual.count == expected.count)
            #expect(zip(actual, expected).allSatisfy { abs($0 - $1) < 1e-4 }, "\(entry.path) fixture \(i)")
            #expect(response.usage.outputTokens == 0)
            #expect(response.usage.inputTokens > 0)
        }
        let request = DecisionRequest(state: "apple", questions: [.noul(id: "q", instructions: "Is this edible?")], metadata: .init(sourceID: "frame", schemaRevision: "1"))
        let response = try await model.predict(request)
        #expect(response.metadata?.sourceID == "frame")
        let sibling = try await model.predict(state: request.state, questions: [
            .noul(id: "one", instructions: "Is this edible?"),
            .noul(id: "two", instructions: "Is this edible?"),
        ])
        #expect(try sibling.noul("one").noul == sibling.noul("two").noul)
        #expect(try sibling.noul("one").noul == response.noul("q").noul)
        let url = URL(fileURLWithPath: entry.path)
        let compiled = try await Task.detached { try MLModel.compileModel(at: url) }.value
        do {
            let compiledModel = try await SwevModel.load(from: compiled, configuration: .init(computeUnits: .cpuOnly))
            let compiledResponse = try await compiledModel.predict(request)
            #expect(try compiledResponse.noul("q").noul == response.noul("q").noul)
        }
        try FileManager.default.removeItem(at: compiled)
        await #expect(throws: SwevError.unsupportedModality) {
            try await model.predict(state: "apple", questions: request.questions, images: [.init(data: Data(), contentType: "image/png")])
        }
        await #expect(throws: SwevError.contextOverflow) {
            try await model.predict(state: .string(String(repeating: "word ", count: 500)), questions: request.questions)
        }
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predict(request)
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
    }
}

@Test func smallTokenizerAndUnsupportedFeatures() throws {
    var root = syntheticTokenizer()
    let tokenizer = try ByteBPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    #expect(try tokenizer.encode("hi!") == [256, 33])
    #expect(try tokenizer.encode("e\u{301}") == [195, 169])
    #expect(try tokenizer.encode("hi  [MASK]!") == [256, 300, 33])
    root["normalizer"] = ["type": "Lowercase"]
    #expect(throws: SwevError.unsupportedTokenizer) { try ByteBPETokenizer(data: JSONSerialization.data(withJSONObject: root)) }
}

@Test func boundedAdmission() throws {
    let admission = RequestAdmission(limit: 2)
    try admission.acquire(); try admission.acquire()
    #expect(throws: SwevError.queueFull) { try admission.acquire() }
    admission.release()
    try admission.acquire()
    admission.release(); admission.release()
}

private func syntheticTokenizer() -> [String: Any] {
    var vocabulary: [String: Int] = [:]
    var extra = 0
    for byte in 0..<256 {
        let visible = (33...126).contains(byte) || (161...172).contains(byte) || (174...255).contains(byte)
        vocabulary[String(UnicodeScalar(visible ? byte : 256 + extra)!)] = byte
        if !visible { extra += 1 }
    }
    vocabulary["hi"] = 256
    let root: [String: Any] = ["version": "1.0", "normalizer": ["type": "NFC"],
        "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false, "use_regex": true],
        "model": ["type": "BPE", "vocab": vocabulary, "merges": [["h", "i"]]],
        "added_tokens": [["id": 300, "content": "[MASK]", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false]]]
    return root
}

@Test func packageSuppliedRecipe() throws {
    let tokenizer = try ByteBPETokenizer(data: JSONSerialization.data(withJSONObject: syntheticTokenizer()))
    let source = #"""
    {"padToken":"[MASK]","state":{"op":"format","value":"text-or-json","args":[{"op":"field","value":"state"}]},
    "instructions":{"op":"literal","value":""},
    "options":{"choice":{"op":"field","value":"id"},"score":{"op":"format","value":"python","args":[{"op":"field","value":"description"}]},"noul":{"op":"indexed","args":[{"op":"literal","value":"N"},{"op":"literal","value":"Y"}]}},
    "replacements":[],"segments":[{"kind":"group","value":"state"},{"kind":"options","segments":[{"kind":"mark"},{"kind":"option"}]}],
    "groupLimits":{"state":8},"scoreLegend":"json"}
    """#
    func adapter(_ json: String) throws -> TextAdapter {
        try TextAdapter(tokenizer: tokenizer, length: 32, optionCapacity: 4,
            recipe: JSONDecoder().decode(TextRecipe.self, from: Data(json.utf8)))
    }
    let question = Question.noul(id: "q", instructions: "ignored")
    let row = try adapter(source).encode(state: "hi", question: question)
    #expect(row.ids == [256, 78, 89])
    #expect(row.options == [1, 2])
    let changed = source.replacingOccurrences(of: #""value":"N""#, with: #""value":"F""#)
    #expect(try adapter(changed).encode(state: "hi", question: question).ids == [256, 70, 89])
    #expect(throws: SwevError.contextOverflow) { try adapter(source).encode(state: "123456789", question: question) }
    #expect(throws: SwevError.invalidMetadata) { try adapter(source.replacingOccurrences(of: #""op":"indexed""#, with: #""op":"script""#)) }
    #expect(throws: SwevError.invalidMetadata) { try adapter(source.replacingOccurrences(of: #"{"kind":"mark"},"#, with: "")) }
}

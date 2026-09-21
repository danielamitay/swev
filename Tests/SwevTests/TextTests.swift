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
    struct Adapter: Decodable {
        let tokenizer: String; let recipe: String; let reference: String
        let sequenceLength: Int?; let optionCapacity: Int?
    }
    let models: [Model]
    struct Model: Decodable { let path: String; let reference: String; let cases: String? }
}
private func manifest() throws -> Manifest {
    try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: ProcessInfo.processInfo.environment["SWEV_TEST_MANIFEST"]!)))
}
private let hasModels = ProcessInfo.processInfo.environment["SWEV_TEST_MANIFEST"] != nil

@Test(.enabled(if: hasModels)) func tokenizerReferenceParity() throws {
    struct Fixture: Decodable { let text: String; let ids: [Int] }
    for entry in try manifest().tokenizers {
        let tokenizer = try BPETokenizer(data: Data(contentsOf: URL(fileURLWithPath: entry.path)))
        let url = URL(fileURLWithPath: entry.reference)
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
        for fixture in fixtures {
            let actual = try tokenizer.encode(fixture.text)
            #expect(actual == fixture.ids, "\(entry.path): \(fixture.text.debugDescription)")
        }
        #expect(throws: SwevError.resourceLimit) { try tokenizer.encode(String(repeating: "x", count: 32769)) }
    }
}

@Test(.enabled(if: hasModels)) func textAdapterReferenceParity() throws {
    let manifest = try manifest()
    for entry in manifest.adapters {
        let tokenizer = try BPETokenizer(data: Data(contentsOf: URL(fileURLWithPath: entry.tokenizer)))
        let recipe = try JSONDecoder().decode(TextRecipe.self, from: Data(contentsOf: URL(fileURLWithPath: entry.recipe)))
        let adapter = try TextAdapter(tokenizer: tokenizer, length: entry.sequenceLength ?? 128,
                                      optionCapacity: entry.optionCapacity ?? recipe.candidateTokens?.count ?? 4, recipe: recipe)
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
        #expect(throws: SwevError.contextOverflow) { try adapter.encode(state: .string(String(repeating: "x ", count: adapter.length + 1)), question: question) }
    }
}

@Test(.enabled(if: hasModels)) func endToEndModels() async throws {
    let fixtureURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("fixtures/text-cases.json")
    for entry in try manifest().models {
        let casesURL = entry.cases.map { URL(fileURLWithPath: $0) } ?? fixtureURL
        guard case .array(let cases) = try JSONValue.parse(Data(contentsOf: casesURL)) else { Issue.record("Invalid fixture"); return }
        let model = try await SwevModel.load(from: URL(fileURLWithPath: entry.path), configuration: .init(computeUnits: .cpuOnly))
        let references = try JSONValue.parse(Data(contentsOf: URL(fileURLWithPath: entry.reference)))
        let rows: [JSONValue]
        if case .array(let items) = references { rows = items }
        else if case .array(let items) = references.member("rows") { rows = items }
        else { Issue.record("Invalid references"); return }
        guard rows.count == cases.count else { Issue.record("Reference count must match text cases"); return }
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
        if model.descriptor.capabilities.supportsImages {
            await #expect(throws: SwevError.self) {
                try await model.predict(state: "apple", questions: request.questions, images: [.init(data: Data(), contentType: "image/png")])
            }
        } else {
            await #expect(throws: SwevError.unsupportedModality) {
                try await model.predict(state: "apple", questions: request.questions, images: [.init(data: Data(), contentType: "image/png")])
            }
        }
        await #expect(throws: SwevError.contextOverflow) {
            try await model.predict(state: .string(String(repeating: "x ", count: model.descriptor.capabilities.limits.maxSequenceTokens + 1)), questions: request.questions)
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
    let tokenizer = try BPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    #expect(try tokenizer.encode("hi!") == [256, 33])
    #expect(try tokenizer.encode("e\u{301}") == [195, 169])
    #expect(try tokenizer.encode("hi  [MASK]!") == [256, 300, 33])
    root["normalizer"] = ["type": "Lowercase"]
    #expect(throws: SwevError.unsupportedTokenizer) { try BPETokenizer(data: JSONSerialization.data(withJSONObject: root)) }
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
    let tokenizer = try BPETokenizer(data: JSONSerialization.data(withJSONObject: syntheticTokenizer()))
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

@Test func unicodeBPEPreservesExactScalars() throws {
    let bytes = (0..<256).map { String(format: "\"<0x%02X>\":%d", $0, $0) }.joined(separator: ",")
    let document = #"""
    {"version":"1.0","normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},
    "pre_tokenizer":{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false},
    "model":{"type":"BPE","byte_fallback":true,"fuse_unk":true,"vocab":{
    BYTE_ENTRIES,"e":300,"é":301,"\u0301":302,"e\u0301":303,"▁":304,"▁\u0301":305,"\ufeff":306,"\ufeff\ufeff":307},
    "merges":[["e","\u0301"],["▁","\u0301"],["\ufeff","\ufeff"]]}}
    """#.replacingOccurrences(of: "BYTE_ENTRIES", with: bytes)
    let tokenizer = try BPETokenizer(data: Data(document.utf8))
    #expect(try tokenizer.encode("é") == [301])
    #expect(try tokenizer.encode("e\u{301}") == [303])
    #expect(try tokenizer.encode(" \u{301}") == [305])
    #expect(try tokenizer.encode("\u{feff}\u{feff}") == [307])
    #expect(try tokenizer.encode("🐈") == [240, 159, 144, 136])
}

@Test func joinedTextAndCandidateTokenIDs() throws {
    let tokenizer = try BPETokenizer(data: JSONSerialization.data(withJSONObject: syntheticTokenizer()))
    let recipe = #"""
    {"padToken":"[MASK]","tokenization":"joined","candidateTokens":["Y","N","A","B"],
    "state":{"op":"field","value":"state"},"instructions":{"op":"literal","value":""},
    "options":{"choice":{"op":"field","value":"id"},"score":{"op":"format","value":"python","args":[{"op":"field","value":"description"}]},"noul":{"op":"indexed","args":[{"op":"literal","value":"i"},{"op":"literal","value":"!"}]}},
    "replacements":[],"segments":[{"kind":"group","value":"state"},{"kind":"options","segments":[{"kind":"option"}]}],"groupLimits":{},"scoreLegend":"json"}
    """#
    func make(_ text: String) throws -> TextAdapter {
        try TextAdapter(tokenizer: tokenizer, length: 32, optionCapacity: 4,
                        recipe: JSONDecoder().decode(TextRecipe.self, from: Data(text.utf8)))
    }
    let row = try make(recipe).encode(state: "h", question: .noul(id: "q", instructions: "unused"))
    #expect(row.ids == [256, 33]) // BPE merges across recipe segment boundaries.
    #expect(row.options == [89, 78]) // Label IDs, not input positions or an assumed ordering.
    #expect(throws: SwevError.invalidMetadata) { try make(recipe.replacingOccurrences(of: #"["Y","N","A","B"]"#, with: #"["multi","N","A","B"]"#)) }
}

@Test func bpeHeapMatchesRankedMergesAndHandlesLongPieces() throws {
    var root = syntheticTokenizer()
    let merges = [["a", "b"], ["b", "a"], ["a", "a"], ["ab", "a"], ["a", "ba"],
                  ["ab", "ab"], ["aa", "b"], ["b", "b"], ["ba", "ba"], ["aba", "b"]]
    var model = root["model"] as! [String: Any]
    var vocabulary = model["vocab"] as! [String: Int]
    for pair in merges {
        let merged = pair.joined()
        if vocabulary[merged] == nil { vocabulary[merged] = vocabulary.count + 1000 }
    }
    model["vocab"] = vocabulary; model["merges"] = merges; root["model"] = model
    let tokenizer = try BPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    func reference(_ text: String) -> [Int] {
        var symbols = text.map(String.init)
        while symbols.count > 1 {
            var best: (Int, Int)?
            for i in 0..<(symbols.count - 1) {
                if let rank = merges.firstIndex(of: [symbols[i], symbols[i + 1]]), rank < (best?.1 ?? Int.max) {
                    best = (i, rank)
                }
            }
            guard let (i, _) = best else { break }
            symbols[i] += symbols.remove(at: i + 1)
        }
        return symbols.map { vocabulary[$0]! }
    }
    var seed: UInt64 = 41
    for length in 0...150 {
        let text = String((0..<length).map { _ -> Character in
            seed = seed &* 6364136223846793005 &+ 1
            return seed >> 61 < 4 ? "a" : "b"
        })
        #expect(try tokenizer.encode(text) == reference(text))
    }
    let long = String(repeating: "ab", count: 8192)
    #expect(try tokenizer.encode(long) == Array(repeating: vocabulary["abab"]!, count: 4096))
}

@Test func codecSupportsSixteenScoreLevels() throws {
    let json = "{\"state\":\"test\",\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"Rate\",\"criteria\":[" + (0..<16).map { "\"level \($0)\"" }.joined(separator: ",") + "]}}}"
    let request = try DecisionCodec.decodeRequest(Data(json.utf8))
    #expect(request.questions[0].optionCount == 16)
}

@Test func identityNormalizationAndIndividualDigits() throws {
    var root = syntheticTokenizer()
    root["normalizer"] = NSNull()
    var model = root["model"] as! [String: Any]
    var vocabulary = model["vocab"] as! [String: Int]
    vocabulary["12"] = 257
    vocabulary["Ġ1"] = 258
    vocabulary["ĠÂ"] = 259
    vocabulary["ĠÂ²"] = 260
    model["vocab"] = vocabulary
    model["merges"] = [["h", "i"], ["1", "2"], ["Ġ", "1"], ["Ġ", "Â"], ["ĠÂ", "²"]]
    root["model"] = model
    let plain = try BPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    #expect(try plain.encode("12") == [257])
    #expect(try plain.encode(" 1") == [258])
    #expect(try plain.encode("e\u{301}") == [101, 204, 129])
    #expect(try plain.encode("é") == [195, 169])
    let byteLevel = root["pre_tokenizer"]!
    root["pre_tokenizer"] = ["type": "Sequence", "pretokenizers": [
        ["type": "Digits", "individual_digits": true], byteLevel
    ]]
    let digits = try BPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    #expect(try digits.encode("hi 12.3") == [256, 32, 49, 50, 46, 51])
    #expect(try digits.encode("١٢") == [217, 161, 217, 162])
    #expect(try digits.encode(" ²") == [260]) // Digits isolates decimal digits, not every numeric category.
    #expect(try digits.encode("hi  [MASK]12") == [256, 300, 49, 50])
    #expect(try digits.encode("e\u{301}") == [101, 204, 129])
    root["pre_tokenizer"] = ["type": "Sequence", "pretokenizers": [
        ["type": "Digits", "individual_digits": false], byteLevel
    ]]
    #expect(throws: SwevError.unsupportedTokenizer) { try BPETokenizer(data: JSONSerialization.data(withJSONObject: root)) }
}

@Test func prunedByteVocabularyRejectsUnrepresentableInput() throws {
    var root = syntheticTokenizer()
    var model = root["model"] as! [String: Any]
    var vocabulary = model["vocab"] as! [String: Int]
    vocabulary.removeValue(forKey: "x")
    model["vocab"] = vocabulary
    root["model"] = model
    let tokenizer = try BPETokenizer(data: JSONSerialization.data(withJSONObject: root))
    #expect(try tokenizer.encode("hi") == [256])
    #expect(throws: SwevError.unsupportedTokenizer) { try tokenizer.encode("x") }
}

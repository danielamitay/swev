import Foundation
import MLX
import MLXDecisionModels
import MLXLMCommon
import Testing
@testable import Swev

@Test func unknownPackedSchemaFailsBeforeModelConstruction() throws {
    let configuration = Data(#"{"model_type":"prism_hadamard_qwen35","schema_version":3}"#.utf8)
    #expect(throws: RotatedQuantization.Failure.self) {
        try RotatedQuantization.factory(configuration: configuration)
    }
}

/// Opt-in: prepared Python reference inputs/logits plus an existing local encoder checkpoint.
@Test func nativeEncoderMatchesReferenceLogits() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let modelPath = environment["SWEV_ENCODER_MODEL"], let fixturePath = environment["SWEV_ENCODER_PARITY"] else { return }
    let directory = URL(fileURLWithPath: modelPath)
    let model = try ModernDecisionEncoder(configuration: Data(contentsOf: directory.appendingPathComponent("encoder/config.json")),
        headLayers: 2, weightsURL: directory.appendingPathComponent("model.safetensors"))
    struct Fixture: Decodable { let tokens: [Int]; let markers: [Int]; let type: Int; let logits: [Float] }
    let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
    let fixtures = try JSONDecoder().decode([Fixture].self, from: data)
    let tokenizerDirectory = directory.appendingPathComponent("tokenizer")
    let tokenizer = try await TokenizerCompatibility().load(from: tokenizerDirectory)
    let configuration = try JSONValue.parse(Data(contentsOf: tokenizerDirectory.appendingPathComponent("tokenizer_config.json")))
    func special(_ key: String) throws -> (String, Int) {
        guard let value = configuration.member(key) else { throw SwevError.unsupportedTokenizer }
        let text: String
        if case .string(let string) = value { text = string }
        else if case .string(let string) = value.member("content") { text = string }
        else { throw SwevError.unsupportedTokenizer }
        return (text, try #require(tokenizer.convertTokenToId(text)))
    }
    let (_, cls) = try special("cls_token"), (_, sep) = try special("sep_token")
    let (maskText, marker) = try special("mask_token")
    let (maximum, head, _, _) = try EncoderDecisionBackend.configuration(directory: directory, limit: nil)
    guard case .array(let rows) = try JSONValue.parse(data) else { throw SwevError.invalidRequest("Expected parity fixtures") }
    for (fixture, row) in zip(fixtures, rows) {
        let request = try DecisionCodec.decodeRequest(Data(try row.jsonString().utf8))
        let prepared = try EncoderDecisionPrompt(state: request.state, question: #require(request.questions.first),
            cls: cls, sep: sep, marker: marker, maskText: maskText, headLimit: head, contextLimit: maximum,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
        #expect(prepared.tokens == fixture.tokens)
        #expect(prepared.markers == fixture.markers)
        #expect(prepared.typeIndex == fixture.type)
        let actual = try model.logits(tokens: fixture.tokens, markers: fixture.markers, questionType: fixture.type)
        #expect(actual.count == fixture.logits.count)
        let difference = zip(actual, fixture.logits).map { abs($0 - $1) }.max() ?? .infinity
        #expect(difference <= 0.03, "Maximum FP16 raw-logit difference: \(difference)")
    }
}

@Test func signedHadamardRoundTripAndNorm() throws {
    guard ProcessInfo.processInfo.environment["SWEV_TEST_MLX"] == "1" else { return }
    let x = MLXArray((0..<1024).map { Float($0 % 17) / 17 }).reshaped([2, 512])
    let signs = MLXArray((0..<512).map { $0 % 3 == 0 ? Float(-1) : Float(1) })
    let transformed = RotatedQuantization.transform(x, block: 512, signs: signs)
    let restored = RotatedQuantization.transform(transformed, block: 512, signs: signs, inverse: true)
    #expect(abs(restored - x).max().item(Float.self) < 1e-5)
    #expect(abs((x * x).sum() - (transformed * transformed).sum()).item(Float.self) < 1e-3)
}

@Test func decisionForwardSupportsFlatLanguageAndBatchedVisionInputs() throws {
    guard ProcessInfo.processInfo.environment["SWEV_TEST_MLX"] == "1" else { return }
    let flat = LMInput.Text(tokens: MLXArray([11, 12, 13]), mask: MLXArray([1, 1, 1]))
    let batched = try flat.batchedForDecisionForward()
    #expect(batched.tokens.shape == [1, 3])
    #expect(batched.mask?.shape == [1, 3])
    #expect(try batched.batchedForDecisionForward().tokens.shape == [1, 3])
    #expect(batched.tokens.reshaped([-1]).asArray(Int.self) == [11, 12, 13])
    #expect(throws: SwevError.inferenceFailed) {
        try LMInput.Text(tokens: MLXArray([1, 2, 3, 4]).reshaped([2, 2])).batchedForDecisionForward()
    }
}

import Foundation
import Testing
@testable import Swev

@Test func orderedJSONAndValidation() throws {
    let state: JSONValue = .object([("z", .array([.null, .bool(true)])), ("a", "café\n🍎")])
    #expect(try state.jsonString() == "{\"z\":[null,true],\"a\":\"café\\n🍎\"}")
    #expect(throws: SwevError.self) { try JSONValue.object([("x", .null), ("x", .null)]).jsonString() }
    #expect(throws: SwevError.self) { try JSONValue.number(.infinity).jsonString() }
    #expect(throws: SwevError.self) { try DecisionRequest(state: .bool(true), questions: [.noul(id: "a", instructions: "Edible?")]).validate() }
}

@Test func requestValidation() throws {
    let question = Question.choice(id: "food", instructions: "Is this edible?", options: [.init(id: "no"), .init(id: "yes")])
    try DecisionRequest(state: "apple", questions: [question]).validate()
    #expect(throws: SwevError.self) { try DecisionRequest(state: "apple", questions: [question, question]).validate() }
    #expect(throws: SwevError.self) { try DecisionRequest(state: "apple", questions: [.choice(id: "a", instructions: "?", options: [.init(id: "x"), .init(id: "x")])]).validate() }
    #expect(throws: SwevError.self) { try DecisionRequest(state: "apple", questions: [.score(id: "a", instructions: "?", levels: ["low"])]).validate() }
}

@Test func distributionsAndTypedAnswers() throws {
    let choice = Question.choice(id: "c", instructions: "Pick", options: [.init(id: "first"), .init(id: "second")])
    let answers = [
        try Postprocessing.answer(question: choice, logits: [10000, 10000]),
        try Postprocessing.answer(question: .noul(id: "n", instructions: "True?"), logits: [0, log(3)]),
        try Postprocessing.answer(question: .score(id: "s", instructions: "Rate", levels: ["low", "medium", "high"]), logits: [0, 0, 0]),
    ]
    let response = DecisionResponse(modelID: "test", modelRevision: "1", answers: answers,
                                    usage: .init(inputTokens: 1, outputTokens: 0), metadata: nil)
    #expect(try response.choice("c").choice == "first")
    #expect(try response.choice("c").confidence.value == 0)
    #expect(abs(try response.noul("n").noul - 0.75) < 1e-12)
    #expect(try response.score("s").score == 1)
    #expect(try response.score("s").modalLevel == 0)
    #expect(throws: SwevError.missingAnswer("absent")) { try response.choice("absent") }
    #expect(throws: SwevError.answerTypeMismatch(id: "n", expected: .choice)) { try response.choice("n") }
}

@Test func rejectsInvalidScoring() {
    for temperature in [0.0, -1, .infinity, .nan] {
        #expect(throws: SwevError.self) { try Postprocessing.probabilities(logits: [0, 1], temperature: temperature) }
    }
    #expect(throws: SwevError.self) { try Postprocessing.probabilities(logits: [0, .nan], temperature: 1) }
}

@Test func confidenceMethods() throws {
    let q = Question.score(id: "s", instructions: "Rate", levels: ["low", "medium", "high"])
    for method in [ConfidenceMethod.chanceAdjustedMax, .modalDistance, .normalizedEntropy] {
        guard case .score(_, let answer) = try Postprocessing.answer(question: q, logits: [-10000, 0, -10000], method: method) else {
            Issue.record("Wrong answer type"); return
        }
        #expect(answer.modalLevel == 1)
        #expect(answer.confidence.value == 1)
        #expect(answer.confidence.method == method)
    }
}

@Test func metadataErrorsAreExplicit() throws {
    #expect(throws: SwevError.missingMetadata(key: "swev.config")) { try ModelDescriptor.read(metadata: [:]) }
    #expect(throws: SwevError.invalidMetadata) { try ModelDescriptor.read(metadata: ["swev.config": "{}"] ) }
    let json = #"{"contractVersion":"2.0","modelVersion":"1","id":"test","architecture":"test","capabilities":{"modalities":["text"],"questionTypes":["noul"],"limits":{"maxQuestionsPerRequest":1,"maxOptionsPerQuestion":2,"maxSequenceTokens":32}},"execution":{"profile":"unknown","inputAdapter":"unknown"}}"#
    #expect(throws: SwevError.unsupportedContractVersion("2.0")) { try ModelDescriptor.read(metadata: ["swev.config": json]) }
}

@Test func invalidModelURL() async {
    await #expect(throws: SwevError.invalidModelAsset) {
        try await SwevModel.load(from: URL(string: "https://example.com/model.mlpackage")!)
    }
}

@Test func futureSchemaIsRejectedBeforeDecodingItsBody() {
    for version in ["1.1", "2.0", "future"] {
        #expect(throws: SwevError.unsupportedContractVersion(version)) {
            try ModelDescriptor.read(metadata: ["swev.config": "{\"contractVersion\":\"\(version)\",\"futureField\":true}"])
        }
    }
}

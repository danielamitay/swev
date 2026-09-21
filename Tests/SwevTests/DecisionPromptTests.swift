import Testing
@testable import Swev

@Test func decisionPromptPreservesCandidateSemantics() throws {
    let choice = Question.choice(id: "internal", instructions: "Choose", options: [
        .init(id: "opaque-id", description: "apple"), .init(id: "stone")
    ])
    let prompt = try DecisionPrompt(state: .object([("item", "fruit")]), question: choice)
    #expect(prompt.labels == ["A", "B"])
    #expect(prompt.text.contains("A. opaque-id: apple\nB. stone"))
    #expect(prompt.text == (try DecisionPrompt(state: .object([("item", "fruit")]), question: choice)).text)
    let score = try DecisionPrompt(state: "x", question: .score(id: "s", instructions: "Rate", levels: ["low", "mid", "high"]))
    #expect(score.text.contains("A. low\nB. mid\nC. high"))
    let noul = try DecisionPrompt(state: "x", question: .noul(id: "n", instructions: "True?"))
    #expect(noul.text.contains("A. No\nB. Yes"))
}

@Test func candidateLabelsRejectAmbiguousTokenization() throws {
    let prompt = try DecisionPrompt(state: "x", question: .noul(id: "n", instructions: "True?"))
    #expect(try prompt.candidateTokenIDs { $0 == "A" ? [42] : [9] } == [42, 9])
    #expect(throws: SwevError.unsupportedTokenizer) { try prompt.candidateTokenIDs { _ in [42] } }
    #expect(throws: SwevError.unsupportedTokenizer) { try prompt.candidateTokenIDs { _ in [1, 2] } }
    #expect(throws: SwevError.unsupportedTokenizer) { try prompt.candidateTokenIDs { _ in [] } }
    #expect(throws: SwevError.tooManyOptions(limit: 26)) {
        try DecisionPrompt(state: "x", question: .score(id: "s", instructions: "Rate", levels: Array(repeating: "level", count: 27)))
    }
}

@Test func missingDescriptionsPreserveLabelsAndBooleanValues() throws {
    let prompt = try DecisionPrompt(state: "x", question: .choice(id: "q", instructions: "Choose", options: [
        .init(id: "none", description: .null), .init(id: "empty", description: ""),
        .init(id: "zero", description: .number(0)), .init(id: "false", description: .bool(false))
    ]))
    #expect(prompt.text.contains("A. none\nB. empty\nC. zero: 0\nD. false: false\n"))
    let noul = try DecisionPrompt(state: "x", question: .noul(id: "n", instructions: "True?",
        falseDescription: .null, trueDescription: ""))
    #expect(noul.text.contains("A. No\nB. Yes\n"))
}

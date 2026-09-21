import Foundation
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


@Test func boundedAdmission() throws {
    let admission = RequestAdmission(limit: 2)
    try admission.acquire(); try admission.acquire()
    #expect(throws: SwevError.queueFull) { try admission.acquire() }
    admission.release()
    try admission.acquire()
    admission.release(); admission.release()
}


@Test func codecSupportsSixteenScoreLevels() throws {
    let json = "{\"state\":\"test\",\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"Rate\",\"criteria\":[" + (0..<16).map { "\"level \($0)\"" }.joined(separator: ",") + "]}}}"
    let request = try DecisionCodec.decodeRequest(Data(json.utf8))
    #expect(request.questions[0].optionCount == 16)
}


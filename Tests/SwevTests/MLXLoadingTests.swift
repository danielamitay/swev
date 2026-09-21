import Foundation
import Testing
@testable import Swev

@Test func mlxLoadingRejectsInvalidLocalSourcesAndAdmission() async {
    do {
        _ = try await SwevModel.load(hf: "unused/model", maxPendingRequests: 0)
        Issue.record("Accepted an unbounded/invalid admission limit")
    } catch { #expect(error as? SwevError == .invalidRequest("Pending request limit must be 1–64")) }
    do {
        _ = try await SwevModel.load(url: URL(fileURLWithPath: "/nonexistent-swev-model-\(UUID().uuidString)"))
        Issue.record("Accepted a missing local model")
    } catch { #expect(error as? SwevError == .invalidModelAsset) }
}

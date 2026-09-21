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

@Test func unsupportedArchitectureFailsBeforeWeightLoading() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try Data(#"{"model_type":"unregistered_test_architecture","max_position_embeddings":512}"#.utf8)
        .write(to: directory.appendingPathComponent("config.json"))
    do {
        _ = try await SwevModel.load(url: directory)
        Issue.record("Accepted an unknown architecture")
    } catch {
        #expect(error as? SwevError == .invalidRequest("Unsupported model_type: unregistered_test_architecture. A matching runtime implementation is required"))
    }
}

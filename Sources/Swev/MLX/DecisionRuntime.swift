import Foundation

/// Common boundary for causal candidate logits and native decision-head runtimes.
protocol DecisionRuntime: Sendable {
    var id: String { get }
    var revision: String { get }
    var maxContextTokens: Int { get }
    var supportsImages: Bool { get }
    func predict(_ request: DecisionRequest) async throws -> DecisionResponse
}

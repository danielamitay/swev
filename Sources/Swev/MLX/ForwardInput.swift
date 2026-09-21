import MLX
import MLXLMCommon

extension LMInput.Text {
    /// Language processors leave the prompt flat for chunked prefill; forwards need a batch axis.
    /// Vision processors already supply one. Preserve their shape instead of adding another.
    func batchedForDecisionForward() throws -> Self {
        guard tokens.size > 0 else { throw SwevError.inferenceFailed }
        switch tokens.ndim {
        case 1:
            return .init(tokens: tokens.expandedDimensions(axis: 0),
                mask: mask.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 })
        case 2 where tokens.dim(0) == 1:
            return self
        default:
            throw SwevError.inferenceFailed
        }
    }
}

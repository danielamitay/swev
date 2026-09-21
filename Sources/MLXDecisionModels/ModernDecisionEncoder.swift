import Foundation
import MLX
import MLXNN

/// Inference-only bidirectional encoder with a typed decision Transformer and marker scorer.
/// Own this object on a serial executor; evaluated logits are its only inference output.
public final class ModernDecisionEncoder {
    public enum Failure: Error { case configuration, weights(String), input }
    private let weights: [String: MLXArray]
    private let width: Int
    private let heads: Int
    private let layers: Int
    private let decisionLayers: Int
    private let window: Int
    private let epsilon: Float
    private let layerTypes: [String]
    private let bases: [String: Float]

    public init(configuration: Data, headLayers: Int, weightsURL: URL) throws {
        guard let c = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
              c["model_type"] as? String == "modernbert",
              c["hidden_activation"] as? String == "gelu",
              let width = c["hidden_size"] as? Int, width > 0,
              let heads = c["num_attention_heads"] as? Int, heads > 0, width % heads == 0, (width / heads) % 2 == 0,
              let layers = c["num_hidden_layers"] as? Int, (1...128).contains(layers),
              let inner = c["intermediate_size"] as? Int, inner > 0,
              let vocab = c["vocab_size"] as? Int, vocab > 0,
              let window = c["local_attention"] as? Int, window > 0,
              (1...16).contains(headLayers), width % max(1, width / 64) == 0,
              c["attention_bias"] as? Bool != true, c["mlp_bias"] as? Bool != true,
              c["norm_bias"] as? Bool != true else { throw Failure.configuration }
        self.width = width; self.heads = heads; self.layers = layers
        self.decisionLayers = headLayers; self.window = window
        epsilon = (c["norm_eps"] as? NSNumber)?.floatValue ?? 1e-5
        guard epsilon.isFinite, epsilon > 0 else { throw Failure.configuration }
        let interval = c["global_attn_every_n_layers"] as? Int ?? 3
        guard interval > 0 else { throw Failure.configuration }
        layerTypes = c["layer_types"] as? [String] ?? (0..<layers).map { $0 % interval == 0 ? "full_attention" : "sliding_attention" }
        guard layerTypes.count == layers, Set(layerTypes).isSubset(of: ["full_attention", "sliding_attention"]) else { throw Failure.configuration }
        let rope = c["rope_parameters"] as? [String: [String: Any]] ?? [:]
        var bases: [String: Float] = [:]
        for (kind, fallback) in [("full_attention", Float(160000)), ("sliding_attention", Float(10000))] {
            guard (rope[kind]?["rope_type"] as? String ?? "default") == "default" else { throw Failure.configuration }
            let key = kind == "full_attention" ? "global_rope_theta" : "local_rope_theta"
            let base = (rope[kind]?["rope_theta"] as? NSNumber)?.floatValue ?? (c[key] as? NSNumber)?.floatValue ?? fallback
            guard base.isFinite, base > 0 else { throw Failure.configuration }
            bases[kind] = base
        }
        self.bases = bases
        weights = try loadArrays(url: weightsURL)
        // Validate every tensor used by inference before any indexed access or matrix operation.
        var expected: [String: [Int]] = ["encoder.embeddings.tok_embeddings.weight": [vocab, width],
            "encoder.embeddings.norm.weight": [width], "encoder.final_norm.weight": [width], "type_emb.weight": [3, width]]
        for i in 0..<layers {
            let p = "encoder.layers.\(i)."
            if i > 0 { expected[p + "attn_norm.weight"] = [width] }
            expected[p + "mlp_norm.weight"] = [width]
            expected[p + "attn.Wqkv.weight"] = [3 * width, width]
            expected[p + "attn.Wo.weight"] = [width, width]
            expected[p + "mlp.Wi.weight"] = [2 * inner, width]
            expected[p + "mlp.Wo.weight"] = [width, inner]
        }
        for i in 0..<headLayers {
            let p = "head.layers.\(i)."
            for norm in ["norm1", "norm2"] {
                expected[p + norm + ".weight"] = [width]; expected[p + norm + ".bias"] = [width]
            }
            for (name, output, input) in [("self_attn.in_proj", 3 * width, width), ("self_attn.out_proj", width, width),
                ("linear1", 4 * width, width), ("linear2", width, 4 * width)] {
                expected[p + name + ".weight"] = [output, input]; expected[p + name + ".bias"] = [output]
            }
        }
        expected["scorer.layers.0.weight"] = [width]; expected["scorer.layers.0.bias"] = [width]
        expected["scorer.layers.1.weight"] = [width, width]; expected["scorer.layers.1.bias"] = [width]
        expected["scorer.layers.3.weight"] = [1, width]; expected["scorer.layers.3.bias"] = [1]
        for (key, shape) in expected {
            guard let tensor = weights[key], tensor.shape == shape,
                  [.float16, .bfloat16, .float32].contains(tensor.dtype) else { throw Failure.weights(key) }
        }
    }

    private func norm(_ x: MLXArray, _ name: String, epsilon: Float) -> MLXArray {
        MLXFast.layerNorm(x, weight: weights[name + ".weight"], bias: weights[name + ".bias"], eps: epsilon)
    }
    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let result = matmul(x, weights[name + ".weight"]!.T)
        return weights[name + ".bias"].map { result + $0 } ?? result
    }
    private func attention(_ x: MLXArray, projection: String, output: String,
                           heads: Int, base: Float?, mask: MLXArray?) -> MLXArray {
        let length = x.dim(1), dimension = width / heads
        let packed = linear(x, projection).reshaped([1, length, 3, heads, dimension])
        var q = packed[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = packed[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = packed[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let base {
            q = MLXFast.RoPE(q, dimensions: dimension, traditional: false, base: base, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: dimension, traditional: false, base: base, scale: 1, offset: 0)
        }
        let result = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(dimension)), mask: mask)
        return linear(result.transposed(0, 2, 1, 3).reshaped([1, length, width]), output)
    }

    /// Returns uncalibrated FP32 logits in the supplied marker order; never generates tokens.
    public func logits(tokens: [Int], markers: [Int], questionType: Int) throws -> [Float] {
        guard !tokens.isEmpty, (0..<3).contains(questionType), markers.count >= 2,
              markers.allSatisfy({ tokens.indices.contains($0) }),
              tokens.allSatisfy({ $0 >= 0 && $0 < weights["encoder.embeddings.tok_embeddings.weight"]!.dim(0) }) else { throw Failure.input }
        var x = weights["encoder.embeddings.tok_embeddings.weight"]![MLXArray(tokens)].expandedDimensions(axis: 0)
        x = norm(x, "encoder.embeddings.norm", epsilon: epsilon)
        let positions = MLXArray(0..<tokens.count)
        let local = (abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)) .<= (window / 2)).reshaped([1, 1, tokens.count, tokens.count])
        for i in 0..<layers {
            let p = "encoder.layers.\(i).", kind = layerTypes[i]
            let normalized = i == 0 ? x : norm(x, p + "attn_norm", epsilon: epsilon)
            x = x + attention(normalized, projection: p + "attn.Wqkv", output: p + "attn.Wo", heads: heads,
                base: bases[kind], mask: kind == "sliding_attention" ? local : nil)
            let pieces = split(linear(norm(x, p + "mlp_norm", epsilon: epsilon), p + "mlp.Wi"), parts: 2, axis: -1)
            x = x + linear(gelu(pieces[0]) * pieces[1], p + "mlp.Wo")
        }
        x = norm(x, "encoder.final_norm", epsilon: epsilon) + weights["type_emb.weight"]![questionType]
        for i in 0..<decisionLayers {
            let p = "head.layers.\(i)."
            x = x + attention(norm(x, p + "norm1", epsilon: 1e-5), projection: p + "self_attn.in_proj",
                output: p + "self_attn.out_proj", heads: max(1, width / 64), base: nil, mask: nil)
            x = x + linear(relu(linear(norm(x, p + "norm2", epsilon: 1e-5), p + "linear1")), p + "linear2")
        }
        let selected = x[0, MLXArray(markers)]
        let scores = linear(gelu(linear(norm(selected, "scorer.layers.0", epsilon: 1e-5), "scorer.layers.1")), "scorer.layers.3").asType(.float32)
        eval(scores)
        return scores.reshaped([-1]).asArray(Float.self)
    }
}

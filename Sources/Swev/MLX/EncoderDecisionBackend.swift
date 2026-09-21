import Foundation
import MLXDecisionModels
import MLXLMCommon

/// Serial decision-head execution with the checkpoint's tokenizer, marker layout, and calibration.
actor EncoderDecisionBackend: DecisionRuntime {
    let id: String
    let revision: String
    let maxContextTokens: Int
    let supportsImages = false
    private let encoder: ModernDecisionEncoder
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let headLimit: Int
    private let temperatures: [Double]
    private let buckets: [String: Double]
    private let cls: Int
    private let sep: Int
    private let marker: Int
    private let maskText: String

    private struct Format: Decodable { let format: String; let format_version: Int }
    private struct AgentConfiguration: Decodable {
        let max_len: Int
        let head_max_len: Int
        let head_layers: Int
        let temperature: [Double]?
        let temperature_by_options: [String: Double]?
    }

    static func configuration(directory: URL, limit: Int?) throws -> (Int, Int, [Double], [String: Double]) {
        let format = try JSONDecoder().decode(Format.self, from: Data(contentsOf: directory.appendingPathComponent("mlx_config.json")))
        guard format.format == "laya-mlx", format.format_version == 1 else {
            throw SwevError.invalidRequest("Unsupported decision-encoder format or version")
        }
        let agent = try JSONDecoder().decode(AgentConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("rl_agent_config.json")))
        let maximum = agent.max_len, head = agent.head_max_len
        guard maximum > 0, head > 4, head < maximum, (1...16).contains(agent.head_layers) else { throw SwevError.invalidModelAsset }
        if let limit, limit <= 0 || limit > maximum {
            throw SwevError.invalidRequest("maxContextTokens cannot exceed the decision encoder's declared max_len")
        }
        let temperatures = agent.temperature ?? [1, 1, 1]
        let buckets = agent.temperature_by_options ?? [:]
        guard temperatures.count == 3, (temperatures + Array(buckets.values)).allSatisfy({ $0.isFinite && $0 > 0 }) else { throw SwevError.invalidModelAsset }
        return (limit ?? maximum, head, temperatures, buckets)
    }

    static func load(directory: URL, id: String, revision: String, contextLimit: Int?) async throws -> EncoderDecisionBackend {
        let (maximum, head, temperatures, buckets) = try configuration(directory: directory, limit: contextLimit)
        let tokenizerDirectory = directory.appendingPathComponent("tokenizer")
        let tokenizer = try await TokenizerCompatibility().load(from: tokenizerDirectory)
        guard let tokens = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))) as? [String: Any] else { throw SwevError.invalidModelAsset }
        func special(_ name: String) throws -> (String, Int) {
            let text = tokens[name] as? String ?? (tokens[name] as? [String: Any])?["content"] as? String
            guard let text, let id = tokenizer.convertTokenToId(text) else { throw SwevError.unsupportedTokenizer }
            return (text, id)
        }
        let (_, cls) = try special("cls_token"), (_, sep) = try special("sep_token"), (maskText, marker) = try special("mask_token")
        let agent = try JSONDecoder().decode(AgentConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("rl_agent_config.json")))
        let encoder = try ModernDecisionEncoder(configuration: Data(contentsOf: directory.appendingPathComponent("encoder/config.json")),
            headLayers: agent.head_layers, weightsURL: directory.appendingPathComponent("model.safetensors"))
        try Task.checkCancellation()
        return EncoderDecisionBackend(id: id, revision: revision, maximum: maximum, head: head, temperatures: temperatures,
            buckets: buckets, tokenizer: tokenizer, cls: cls, sep: sep, marker: marker, maskText: maskText, encoder: encoder)
    }

    private init(id: String, revision: String, maximum: Int, head: Int, temperatures: [Double], buckets: [String: Double],
                 tokenizer: any MLXLMCommon.Tokenizer, cls: Int, sep: Int, marker: Int, maskText: String, encoder: sending ModernDecisionEncoder) {
        self.id = id; self.revision = revision; maxContextTokens = maximum; headLimit = head
        self.temperatures = temperatures; self.buckets = buckets; self.tokenizer = tokenizer
        self.cls = cls; self.sep = sep; self.marker = marker; self.maskText = maskText; self.encoder = encoder
    }

    func predict(_ request: DecisionRequest) async throws -> DecisionResponse {
        try request.validate()
        guard request.images.isEmpty else { throw SwevError.unsupportedModality }
        var answers: [Answer] = [], count = 0
        for question in request.questions {
            try Task.checkCancellation()
            guard question.optionCount <= 26 else { throw SwevError.tooManyOptions(limit: 26) }
            let prepared = try EncoderDecisionPrompt(state: request.state, question: question, cls: cls, sep: sep, marker: marker,
                maskText: maskText, headLimit: headLimit, contextLimit: maxContextTokens,
                encode: { tokenizer.encode(text: $0, addSpecialTokens: false) })
            let raw = try encoder.logits(tokens: prepared.tokens, markers: prepared.markers, questionType: prepared.typeIndex)
            try Task.checkCancellation()
            let size = question.optionCount <= 2 ? "2" : question.optionCount <= 5 ? "3-5" : question.optionCount <= 10 ? "6-10" : "11+"
            let temperature = buckets[question.type.rawValue + ":" + size] ?? temperatures[prepared.typeIndex]
            answers.append(try Postprocessing.answer(question: question, logits: raw.map(Double.init), temperature: temperature))
            count += prepared.tokens.count
        }
        return DecisionResponse(modelID: id, modelRevision: revision, answers: answers,
            usage: .init(inputTokens: count, outputTokens: 0), metadata: request.metadata)
    }
}

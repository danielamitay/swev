import Foundation

/// Completes declared assistant headers before adding decision content.
/// Detection uses tokenizer control tokens, never a repository or architecture name.
struct AssistantContinuation: Sendable {
    private let plain: [Int]
    private let openHeader: [Int]?
    private let finalHeader: [Int]?
    private let finalContent: [Int]?

    static let probe = "SWEV_ASSISTANT_CONTENT_PROBE"

    init(encode: (String) -> [Int], decode: ([Int]) -> String,
         renderAssistant: (() throws -> String)? = nil) throws {
        plain = encode(DecisionPrompt.answerPrefix)
        func isControl(_ text: String) -> Bool {
            let tokens = encode(text)
            return tokens.count == 1 && decode(tokens) == text
        }
        // Both spellings are used by channel-based chat templates/tokenizers.
        let convention = [("<|start|>", "<|channel|>", "<|message|>"),
                          ("<|im_start|>", "<|meta_sep|>", "<|im_sep|>")]
            .first { isControl($0.0) && isControl($0.1) && isControl($0.2) }
        if let (start, channel, message) = convention {
            openHeader = encode(start + "assistant")
            finalHeader = encode(start + "assistant" + channel + "final" + message)
            finalContent = encode(channel + "final" + message + DecisionPrompt.answerPrefix)
        } else if isControl("<|start|>"), isControl("<|message|>"), let renderAssistant {
            // Read recipient syntax from a rendered assistant message, not a model-name rule.
            let rendered = try renderAssistant()
            guard let content = rendered.range(of: Self.probe),
                  let start = rendered[..<content.lowerBound].range(of: "<|start|>assistant", options: .backwards) else {
                throw SwevError.invalidRequest("Cannot identify the template’s assistant content header")
            }
            let header = String(rendered[start.lowerBound..<content.lowerBound])
            guard header.hasSuffix("<|message|>"),
                  !header.dropFirst("<|start|>assistant".count).contains("<|start|>") else {
                throw SwevError.invalidRequest("Unsupported assistant content header")
            }
            openHeader = encode("<|start|>assistant")
            finalHeader = encode(header)
            finalContent = encode(String(header.dropFirst("<|start|>assistant".count)) + DecisionPrompt.answerPrefix)
        } else {
            openHeader = nil
            finalHeader = nil
            finalContent = nil
        }
    }

    /// Only this many trailing prompt tokens need inspection; ordinary templates need none.
    var tailCount: Int { finalHeader?.count ?? 0 }

    func tokens(after tail: [Int]) throws -> [Int] {
        guard let openHeader, let finalHeader, let finalContent else { return plain }
        if tail.suffix(openHeader.count).elementsEqual(openHeader) { return finalContent }
        if tail.suffix(finalHeader.count).elementsEqual(finalHeader) { return plain }
        throw SwevError.invalidRequest("Chat template must end in an open assistant header or its declared content header")
    }
}

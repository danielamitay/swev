import Foundation

/// Byte-level or Unicode BPE configured by the embedded tokenizer document.
/// Decoding and tokenizer-supplied postprocessors are deliberately not executed.
struct BPETokenizer {
    private struct Pair: Hashable { let left: Data; let right: Data }
    private struct Added: Decodable {
        let id: Int
        let content: String
        let single_word: Bool
        let lstrip: Bool
        let rstrip: Bool
        let normalized: Bool
    }
    private let vocabulary: [Data: Int]
    private let ranks: [Pair: Int]
    private let added: [Added]
    private let rawAdded: NSRegularExpression?
    private let normalizedAdded: NSRegularExpression?
    private let addedByText: [String: Added]
    private let split: NSRegularExpression?
    private let spaceMarker: String?
    private let bytes: [String]

    static let byteLevelPattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              root["version"] as? String == "1.0",
              let normalizer = root["normalizer"] as? [String: Any],
              let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
              let rawVocabulary = model["vocab"] as? NSDictionary,
              let sourceMerges = model["merges"] as? [Any],
              let pre = root["pre_tokenizer"] as? [String: Any] else { throw SwevError.unsupportedTokenizer }
        var merges = sourceMerges
        var vocabulary: [Data: Int] = [:]
        if normalizer["type"] as? String == "Replace" {
            // Foundation dictionaries can collapse distinct Unicode spellings in vocabularies.
            let document = try JSONValue.parse(data, maximumBytes: 32 * 1024 * 1024)
            guard case .object(let entries) = document.member("model")?.member("vocab") else { throw SwevError.unsupportedTokenizer }
            guard case .array(let mergeValues) = document.member("model")?.member("merges") else { throw SwevError.unsupportedTokenizer }
            merges = try mergeValues.map { value -> [String] in
                guard case .array(let parts) = value else { throw SwevError.unsupportedTokenizer }
                return try parts.map { part in
                    guard case .string(let text) = part else { throw SwevError.unsupportedTokenizer }
                    return text
                }
            }
            for (key, value) in entries {
                guard case .number(let number) = value, number >= 0, number <= Double(Int32.max), number.rounded() == number else { throw SwevError.unsupportedTokenizer }
                vocabulary[Data(key.utf8)] = Int(number)
            }
        } else {
            for (key, value) in rawVocabulary {
                guard let key = key as? String, let value = value as? Int else { throw SwevError.unsupportedTokenizer }
                vocabulary[Data(key.utf8)] = value
            }
        }
        let marker: String?
        if normalizer["type"] as? String == "NFC" {
            marker = nil
            guard model["unk_token"] == nil || model["unk_token"] is NSNull else { throw SwevError.unsupportedTokenizer }
            for key in ["fuse_unk", "byte_fallback"] {
                guard model[key] == nil || model[key] as? Bool == false else { throw SwevError.unsupportedTokenizer }
            }
        } else {
            guard normalizer["type"] as? String == "Replace", (normalizer["pattern"] as? [String: String])?["String"] == " ",
                  let replacement = normalizer["content"] as? String, replacement.unicodeScalars.count == 1, replacement != " ",
                  model["byte_fallback"] as? Bool == true, model["fuse_unk"] as? Bool == true,
                  pre["type"] as? String == "Split", (pre["pattern"] as? [String: String])?["String"] == " ",
                  pre["behavior"] as? String == "MergedWithPrevious", pre["invert"] as? Bool == false else { throw SwevError.unsupportedTokenizer }
            marker = replacement
        }
        guard model["dropout"] == nil || model["dropout"] is NSNull else { throw SwevError.unsupportedTokenizer }
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            guard model[key] == nil || model[key] is NSNull || model[key] as? String == "" else { throw SwevError.unsupportedTokenizer }
        }
        for key in ["ignore_merges"] {
            guard model[key] == nil || model[key] as? Bool == false else { throw SwevError.unsupportedTokenizer }
        }
        func byteLevel(_ value: [String: Any], regex: Bool) -> Bool {
            value["type"] as? String == "ByteLevel" && value["add_prefix_space"] as? Bool == false && value["use_regex"] as? Bool == regex
        }
        let pattern: String?
        if marker != nil {
            pattern = nil
        } else if byteLevel(pre, regex: true) {
            pattern = Self.byteLevelPattern
        } else {
            guard pre["type"] as? String == "Sequence", let stages = pre["pretokenizers"] as? [[String: Any]], stages.count == 2,
                  stages[0]["type"] as? String == "Split", stages[0]["behavior"] as? String == "Isolated",
                  stages[0]["invert"] as? Bool == false,
                  let configured = (stages[0]["pattern"] as? [String: String])?["Regex"], configured.utf8.count <= 2048,
                  byteLevel(stages[1], regex: false) else { throw SwevError.unsupportedTokenizer }
            pattern = configured
        }
        var ranks: [Pair: Int] = [:]
        for (rank, merge) in merges.enumerated() {
            let parts = (merge as? [String]) ?? (merge as? String)?.components(separatedBy: " ") ?? []
            guard parts.count == 2, vocabulary[Data((parts[0] + parts[1]).utf8)] != nil,
                  ranks.updateValue(rank, forKey: Pair(left: Data(parts[0].utf8), right: Data(parts[1].utf8))) == nil else {
                throw SwevError.unsupportedTokenizer
            }
        }
        let addedData = try JSONSerialization.data(withJSONObject: root["added_tokens"] ?? [])
        let added = try JSONDecoder().decode([Added].self, from: addedData)
        guard added.allSatisfy({ !$0.content.isEmpty && !$0.single_word && $0.id >= 0 && $0.id <= Int(Int32.max) }),
              Set(added.map(\.content)).count == added.count,
              vocabulary.values.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }) else { throw SwevError.unsupportedTokenizer }
        func regex(_ entries: [Added]) throws -> NSRegularExpression? {
            guard !entries.isEmpty else { return nil }
            return try NSRegularExpression(pattern: entries.sorted { $0.content.utf8.count > $1.content.utf8.count }
                .map { NSRegularExpression.escapedPattern(for: $0.content) }.joined(separator: "|"))
        }
        self.vocabulary = vocabulary
        self.ranks = ranks
        self.added = added
        self.addedByText = Dictionary(uniqueKeysWithValues: added.map { ($0.content, $0) })
        self.rawAdded = try regex(added.filter { !$0.normalized })
        self.normalizedAdded = try regex(added.filter(\.normalized))
        self.split = try pattern.map { try NSRegularExpression(pattern: $0) }
        self.spaceMarker = marker
        var mapping = Array(repeating: "", count: 256)
        var extra = 0
        for byte in 0..<256 {
            let visible = (33...126).contains(byte) || (161...172).contains(byte) || (174...255).contains(byte)
            mapping[byte] = String(UnicodeScalar(visible ? byte : 256 + extra)!)
            if !visible { extra += 1 }
        }
        if marker == nil {
            guard mapping.enumerated().allSatisfy({ [192, 193].contains($0.offset) || $0.offset >= 245 || vocabulary[Data($0.element.utf8)] != nil }) else { throw SwevError.unsupportedTokenizer }
        } else {
            guard (0..<256).allSatisfy({ vocabulary[Data(String(format: "<0x%02X>", $0).utf8)] != nil }) else { throw SwevError.unsupportedTokenizer }
        }
        self.bytes = mapping
    }

    func tokenID(_ text: String) throws -> Int {
        guard let id = addedByText[text]?.id ?? vocabulary[Data(text.utf8)] else { throw SwevError.unsupportedTokenizer }
        return id
    }

    func encode(_ text: String) throws -> [Int] {
        guard text.utf8.count <= 32_768 else { throw SwevError.resourceLimit }
        return try extract(text, regex: rawAdded) { raw in
            let normalized: String
            if let marker = spaceMarker {
                normalized = raw.unicodeScalars.map { scalar -> String in scalar.value == 32 ? marker : String(scalar) }.joined()
            } else {
                normalized = raw.precomposedStringWithCanonicalMapping
            }
            return try extract(normalized, regex: normalizedAdded, plain: encodePlain)
        }
    }

    private func extract(_ text: String, regex: NSRegularExpression?, plain: (String) throws -> [Int]) throws -> [Int] {
        guard let regex else { return try plain(text) }
        let string = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: string.length))
        var offset = 0
        var result: [Int] = []
        for match in matches {
            let token = addedByText[string.substring(with: match.range)]!
            var start = match.range.location
            var end = NSMaxRange(match.range)
            if token.lstrip {
                while start > offset, let scalar = UnicodeScalar(string.character(at: start - 1)), CharacterSet.whitespacesAndNewlines.contains(scalar) { start -= 1 }
            }
            if token.rstrip {
                while end < string.length, let scalar = UnicodeScalar(string.character(at: end)), CharacterSet.whitespacesAndNewlines.contains(scalar) { end += 1 }
            }
            guard start >= offset else { continue }
            result += try plain(string.substring(with: NSRange(location: offset, length: start - offset)))
            result.append(token.id)
            offset = end
        }
        result += try plain(string.substring(from: offset))
        return result
    }

    private func encodePlain(_ text: String) throws -> [Int] {
        let string = text as NSString
        var result: [Int] = []
        var offset = 0
        let ranges = split?.matches(in: text, range: NSRange(location: 0, length: string.length)).map(\.range) ?? [NSRange(location: 0, length: string.length)]
        for range in ranges {
            guard range.location == offset else { throw SwevError.unsupportedTokenizer }
            offset = NSMaxRange(range)
            let piece = string.substring(with: range)
            var symbols = spaceMarker == nil ? piece.utf8.map { bytes[Int($0)] } : piece.unicodeScalars.map(String.init)
            guard symbols.count <= 4096 else { throw SwevError.resourceLimit }
            while symbols.count > 1 {
                var best: (index: Int, rank: Int)?
                for i in 0..<(symbols.count - 1) {
                    if let rank = ranks[Pair(left: Data(symbols[i].utf8), right: Data(symbols[i + 1].utf8))], rank < (best?.rank ?? Int.max) {
                        best = (i, rank)
                    }
                }
                guard let best else { break }
                symbols[best.index] += symbols[best.index + 1]
                symbols.remove(at: best.index + 1)
            }
            for symbol in symbols {
                if let id = vocabulary[Data(symbol.utf8)] { result.append(id) }
                else if spaceMarker != nil {
                    for byte in symbol.utf8 { result.append(try tokenID(String(format: "<0x%02X>", Int(byte)))) }
                } else { throw SwevError.unsupportedTokenizer }
            }
        }
        guard offset == string.length else { throw SwevError.unsupportedTokenizer }
        return result
    }
}

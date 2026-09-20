import Foundation

struct EncodedQuestion {
    let ids: [Int]
    let options: [Int]
    let type: Int
    var count: Int { ids.count }
}

/// A bounded, declarative text recipe supplied by the model package.
struct TextRecipe: Decodable {
    struct Expression: Decodable {
        let op: String
        var value: String? = nil
        var args: [Expression]? = nil
    }
    struct Replacement: Decodable { let pattern: String; let replacement: String }
    struct Segment: Decodable {
        let kind: String
        var value: String? = nil
        var segments: [Segment]? = nil
    }
    struct Budget: Decodable {
        let maximum: Int
        let minimumInstructionSlots: Int
        let optionOverhead: Int
    }
    var candidateTokens: [String]? = nil
    var tokenization: String? = nil
    let padToken: String
    let state: Expression
    let instructions: Expression
    let options: [String: Expression]
    let replacements: [Replacement]
    let segments: [Segment]
    let groupLimits: [String: Int]
    let prefixBudget: Budget?
    let scoreLegend: String
}

struct TextAdapter {
    let tokenizer: BPETokenizer
    let length: Int
    let optionCapacity: Int
    let recipe: TextRecipe
    let padID: Int

    init(tokenizer: BPETokenizer, length: Int, optionCapacity: Int, recipe: TextRecipe) throws {
        self.tokenizer = tokenizer
        self.length = length
        self.optionCapacity = optionCapacity
        self.recipe = recipe
        self.padID = try tokenizer.tokenID(recipe.padToken)
        guard recipe.tokenization == nil || ["segments", "joined"].contains(recipe.tokenization) else { throw SwevError.invalidMetadata }
        if let labels = recipe.candidateTokens {
            guard recipe.tokenization == "joined", labels.count == optionCapacity, Set(labels).count == labels.count else { throw SwevError.invalidMetadata }
            let ids = try labels.map { label in
                let tokens = try tokenizer.encode(label)
                guard tokens.count == 1 else { throw SwevError.invalidMetadata }
                return tokens[0]
            }
            guard Set(ids).count == ids.count else { throw SwevError.invalidMetadata }
        } else if recipe.tokenization == "joined" { throw SwevError.invalidMetadata }
        guard Set(recipe.options.keys) == Set(["choice", "score", "noul"]),
              ["json", "indented"].contains(recipe.scoreLegend),
              Set(recipe.groupLimits.keys).isSubset(of: ["state", "instructions", "options"]),
              recipe.groupLimits.values.allSatisfy({ (1...32768).contains($0) }), recipe.replacements.count <= 16 else { throw SwevError.invalidMetadata }
        if let budget = recipe.prefixBudget {
            guard (1...32768).contains(budget.maximum), (0...32768).contains(budget.minimumInstructionSlots),
                  (0...32768).contains(budget.optionOverhead) else { throw SwevError.invalidMetadata }
        }
        func validate(_ expression: TextRecipe.Expression, depth: Int = 0) throws {
            guard depth < 16 else { throw SwevError.invalidMetadata }
            let args = expression.args ?? []
            let count = args.count
            switch expression.op {
            case "literal": guard count == 0, let text = expression.value, text.utf8.count <= 32768 else { throw SwevError.invalidMetadata }
            case "field": guard count == 0, ["state", "instructions", "type", "id", "description", "index"].contains(expression.value) else { throw SwevError.invalidMetadata }
            case "concat", "indexed": guard (1...32).contains(count) else { throw SwevError.invalidMetadata }
            case "truthy", "present": guard count == 3 else { throw SwevError.invalidMetadata }
            case "format": guard count == 1, ["python", "indented", "text-or-json", "string"].contains(expression.value) else { throw SwevError.invalidMetadata }
            default: throw SwevError.invalidMetadata
            }
            for arg in args { try validate(arg, depth: depth + 1) }
        }
        for expression in [recipe.state, recipe.instructions] + Array(recipe.options.values) { try validate(expression) }
        for rule in recipe.replacements {
            guard rule.pattern.utf8.count <= 2048, rule.replacement.utf8.count <= 2048 else { throw SwevError.invalidMetadata }
            _ = try NSRegularExpression(pattern: rule.pattern)
        }
        var optionBlocks = 0
        func validateSegments(_ segments: [TextRecipe.Segment], inOption: Bool = false) throws {
            guard !segments.isEmpty, segments.count <= 64 else { throw SwevError.invalidMetadata }
            var marks = 0
            for segment in segments {
                if segment.kind != "options", segment.segments != nil { throw SwevError.invalidMetadata }
                switch segment.kind {
                case "text": guard recipe.tokenization == "joined", let text = segment.value, text.utf8.count <= 32768 else { throw SwevError.invalidMetadata }
                case "token": guard let token = segment.value else { throw SwevError.invalidMetadata }; _ = try tokenizer.tokenID(token)
                case "group": guard ["state", "instructions"].contains(segment.value) else { throw SwevError.invalidMetadata }
                case "options":
                    guard !inOption, let children = segment.segments else { throw SwevError.invalidMetadata }
                    optionBlocks += 1
                    try validateSegments(children, inOption: true)
                case "option": guard inOption else { throw SwevError.invalidMetadata }
                case "mark":
                    guard inOption, segment.value == nil || segment.value == "previous" else { throw SwevError.invalidMetadata }
                    marks += 1
                default: throw SwevError.invalidMetadata
                }
            }
            guard !inOption || marks == (recipe.candidateTokens == nil ? 1 : 0) else { throw SwevError.invalidMetadata }
        }
        try validateSegments(recipe.segments)
        guard optionBlocks == 1 else { throw SwevError.invalidMetadata }
    }

    func encode(state: JSONValue, question: Question) throws -> EncodedQuestion {
        guard question.optionCount <= optionCapacity else { throw SwevError.tooManyOptions(limit: optionCapacity) }
        let base: [String: JSONValue] = ["state": state, "instructions": question.instructions, "type": .string(question.type.rawValue)]
        func sanitize(_ text: String) throws -> String {
            var text = text
            for rule in recipe.replacements {
                let regex = try NSRegularExpression(pattern: rule.pattern)
                text = regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: rule.replacement)
            }
            return text
        }
        func render(_ expression: TextRecipe.Expression, _ fields: [String: JSONValue]) throws -> String {
            let value = try evaluate(expression, fields: fields)
            guard case .string(let text) = value else { throw SwevError.invalidMetadata }
            return try sanitize(text)
        }
        var optionFields: [[String: JSONValue]] = []
        switch question {
        case .choice(_, _, let options):
            optionFields = options.enumerated().map { ["id": .string($0.element.id), "description": $0.element.description ?? .null, "index": .number(Double($0.offset))] }
        case .score(_, _, let levels):
            optionFields = levels.enumerated().map { ["index": .number(Double($0.offset)), "description": $0.element] }
        case .noul(_, _, let no, let yes):
            optionFields = [["index": .number(0), "description": no ?? .null], ["index": .number(1), "description": yes ?? .null]]
        }
        let optionExpression = recipe.options[question.type.rawValue]!
        let optionTexts = try optionFields.map { try render(optionExpression, base.merging($0) { _, new in new }) }
        let textGroups = try ["state": render(recipe.state, base), "instructions": render(recipe.instructions, base)]
        let options = try optionTexts.map(tokenizer.encode)
        let groups = try textGroups.mapValues(tokenizer.encode)
        for (name, limit) in recipe.groupLimits {
            let rows = name == "options" ? options : [groups[name] ?? []]
            guard rows.allSatisfy({ $0.count <= limit }) else { throw SwevError.contextOverflow }
        }
        if let budget = recipe.prefixBudget {
            let remaining = budget.maximum - options.reduce(0) { $0 + $1.count + budget.optionOverhead }
            guard remaining >= budget.minimumInstructionSlots, groups["instructions"]!.count <= remaining else { throw SwevError.contextOverflow }
        }
        if recipe.tokenization == "joined" {
            func join(_ segments: [TextRecipe.Segment], option: String? = nil) throws -> String {
                var text = ""
                for segment in segments {
                    switch segment.kind {
                    case "token", "text": text += segment.value!
                    case "group": text += textGroups[segment.value!]!
                    case "options":
                        for value in optionTexts { text += try join(segment.segments!, option: value) }
                    case "option": text += option!
                    default: throw SwevError.invalidMetadata
                    }
                    guard text.utf8.count <= 32768 else { throw SwevError.resourceLimit }
                }
                return text
            }
            let ids = try tokenizer.encode(join(recipe.segments))
            guard !ids.isEmpty, ids.count <= length else { throw SwevError.contextOverflow }
            let labels = try recipe.candidateTokens!.prefix(question.optionCount).map { try tokenizer.encode($0)[0] }
            return .init(ids: ids, options: labels, type: question.type == .choice ? 0 : question.type == .score ? 1 : 2)
        }
        var ids: [Int] = [], positions: [Int] = []
        func emit(_ segments: [TextRecipe.Segment], option: [Int]? = nil, depth: Int = 0) throws {
            guard depth < 8, segments.count <= 64 else { throw SwevError.invalidMetadata }
            for segment in segments {
                switch segment.kind {
                case "token":
                    guard let value = segment.value else { throw SwevError.invalidMetadata }
                    ids.append(try tokenizer.tokenID(value))
                case "group":
                    guard let value = segment.value, let group = groups[value] else { throw SwevError.invalidMetadata }
                    ids += group
                case "options":
                    guard option == nil, let children = segment.segments else { throw SwevError.invalidMetadata }
                    for row in options { try emit(children, option: row, depth: depth + 1) }
                case "option":
                    guard let option else { throw SwevError.invalidMetadata }
                    ids += option
                case "mark":
                    guard option != nil else { throw SwevError.invalidMetadata }
                    positions.append(ids.count + (segment.value == "previous" ? -1 : 0))
                default: throw SwevError.invalidMetadata
                }
                guard ids.count <= length else { throw SwevError.contextOverflow }
            }
        }
        try emit(recipe.segments)
        guard positions.count == question.optionCount, positions.allSatisfy({ ids.indices.contains($0) }) else { throw SwevError.invalidMetadata }
        return .init(ids: ids, options: positions, type: question.type == .choice ? 0 : question.type == .score ? 1 : 2)
    }

    private func evaluate(_ expression: TextRecipe.Expression, fields: [String: JSONValue], depth: Int = 0) throws -> JSONValue {
        guard depth < 16, (expression.args?.count ?? 0) <= 32 else { throw SwevError.invalidMetadata }
        let args = expression.args ?? []
        func arg(_ index: Int) throws -> JSONValue {
            guard args.indices.contains(index) else { throw SwevError.invalidMetadata }
            return try evaluate(args[index], fields: fields, depth: depth + 1)
        }
        switch expression.op {
        case "literal": return .string(expression.value ?? "")
        case "field": return fields[expression.value ?? ""] ?? .null
        case "concat":
            var result = ""
            for i in args.indices {
                guard case .string(let value) = try arg(i) else { throw SwevError.invalidMetadata }
                result += value
                guard result.utf8.count <= 32768 else { throw SwevError.resourceLimit }
            }
            return .string(result)
        case "truthy": return try arg(0).isTruthy ? arg(1) : arg(2)
        case "present":
            let value = try arg(0)
            return try !value.isNull && !value.isEmptyString ? arg(1) : arg(2)
        case "indexed":
            guard case .number(let index) = fields["index"], index >= 0, index < Double(args.count) else { throw SwevError.invalidMetadata }
            return try arg(Int(index))
        case "format":
            let value = try arg(0)
            switch expression.value {
            case "python": return .string(try pythonString(value))
            case "indented": return .string(try value.indentedText())
            case "text-or-json":
                if case .string = value { return value }
                return .string(try value.spacedJSON())
            case "string":
                guard case .string = value else { throw SwevError.invalidRequest("This model requires a string description") }
                return value
            default: throw SwevError.invalidMetadata
            }
        default: throw SwevError.invalidMetadata
        }
    }
}

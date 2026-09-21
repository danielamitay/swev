import Foundation

/// Backend-independent candidate normalization and typed decision semantics.
enum Postprocessing {
    static func probabilities(logits: [Double], temperature: Double) throws -> [Double] {
        guard logits.count >= 2, temperature.isFinite, temperature > 0, logits.allSatisfy(\.isFinite) else {
            throw SwevError.nonFiniteOutput
        }
        let maximum = logits.max()!
        let weights = logits.map { exp(($0 - maximum) / temperature) }
        let sum = weights.reduce(0, +)
        return weights.map { $0 / sum }
    }

    static func answer(question: Question, logits: [Double], temperature: Double = 1,
                       method: ConfidenceMethod = .normalizedEntropy) throws -> Answer {
        guard logits.count == question.optionCount else { throw SwevError.invalidRequest("Logit count mismatch") }
        let p = try probabilities(logits: logits, temperature: temperature)
        let mode = p.indices.max { p[$0] < p[$1] }!
        let confidence: Double
        switch method {
        case .normalizedEntropy:
            confidence = 1 + p.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) } / log(Double(p.count))
        case .chanceAdjustedMax:
            let chance = 1 / Double(p.count)
            confidence = (p[mode] - chance) / (1 - chance)
        case .modalDistance:
            confidence = 1 - p.enumerated().reduce(0) { $0 + $1.element * Double(abs($1.offset - mode)) } / Double(p.count - 1)
        }
        let statistic = Confidence(value: min(1, max(0, confidence)), method: method)
        switch question {
        case .choice(let id, _, let options):
            return .choice(id: id, .init(choice: options[mode].id,
                probabilities: zip(options, p).map { .init(id: $0.id, probability: $1) },
                selectedProbability: p[mode], confidence: statistic))
        case .noul(let id, _, _, _): return .noul(id: id, .init(noul: p[1]))
        case .score(let id, _, let levels):
            let legend = try levels.map { value in
                if case .string(let text) = value { return text }
                return try value.jsonString()
            }
            return .score(id: id, .init(score: p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element },
                modalLevel: mode, probabilities: p, legend: legend, confidence: statistic))
        }
    }
}

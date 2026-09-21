import Foundation
import Swev

@main struct Runner {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 3 else {
            throw SwevError.invalidRequest("Usage: SwevBenchmark MODEL_ID_OR_DIRECTORY CASES.jsonl")
        }
        let clock = ContinuousClock()
        func elapsed(_ start: ContinuousClock.Instant) -> Double {
            let c = start.duration(to: clock.now).components
            return Double(c.seconds) + Double(c.attoseconds) / 1e18
        }
        func emit(_ record: [String: Any]) throws {
            FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) + Data([10]))
        }
        let start = clock.now
        let location = URL(fileURLWithPath: args[1])
        let model: SwevModel
        if FileManager.default.fileExists(atPath: location.path) || args[1].hasPrefix("/") || args[1].hasPrefix(".") {
            model = try await SwevModel.load(url: location)
        } else {
            model = try await SwevModel.load(hf: args[1])
        }
        try emit(["phase": "loaded", "model": model.descriptor.id, "load_seconds": elapsed(start)])
        let rows = try String(contentsOfFile: args[2], encoding: .utf8).split(separator: "\n")
        for line in rows {
            let row = try JSONValue.parse(Data(line.utf8))
            guard case .string(let id) = row.field("id"), let raw = row.field("request") else { throw SwevError.invalidRequest("Each case needs id and request fields") }
            let decoded = try DecisionCodec.decodeRequest(Data(try raw.jsonString().utf8))
            var images: [ImageInput] = []
            if case .string(let path) = row.field("image") {
                images = [.init(data: try Data(contentsOf: URL(fileURLWithPath: path)), contentType: "image/png")]
            }
            let request = DecisionRequest(state: decoded.state, questions: decoded.questions, images: images, metadata: decoded.metadata)
            do {
                _ = try await model.predict(request)
                let start = clock.now
                let response = try await model.predict(request)
                let duration = elapsed(start)
                try emit(["phase": "prediction", "task_id": id, "ok": true, "latency_s": duration,
                    "response": try JSONSerialization.jsonObject(with: DecisionCodec.encodeResponse(response))])
            } catch {
                try emit(["phase": "prediction", "task_id": id, "ok": false, "error": String(describing: error)])
            }
        }
        try emit(["phase": "done", "count": rows.count])
    }
}

private extension JSONValue {
    func field(_ name: String) -> JSONValue? {
        guard case .object(let fields) = self else { return nil }
        return fields.first { $0.0 == name }?.1
    }
}

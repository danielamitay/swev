import Foundation
import Testing
@testable import Swev

@Test func contextLimitsRespectDeclarationsAndExplicitBounds() throws {
    let data = Data(#"{"model_type":"arbitrary_architecture","max_position_embeddings":100,"text_config":{"max_position_embeddings":80}}"#.utf8)
    #expect(try ModelConfigurationInspection(data: data).maxContextTokens == 80)
    #expect(try ModelConfigurationInspection(data: data, contextLimit: 40).maxContextTokens == 40)
    #expect(throws: SwevError.self) { try ModelConfigurationInspection(data: data, contextLimit: 81) }
    let absent = Data(#"{"model_type":"arbitrary_architecture"}"#.utf8)
    #expect(throws: SwevError.self) { try ModelConfigurationInspection(data: absent) }
    #expect(try ModelConfigurationInspection(data: absent, contextLimit: 4096).maxContextTokens == 4096)
    for invalid in ["true", "0", "-1", "1.5", "\"8192\""] {
        let data = Data("{\"model_type\":\"test\",\"max_position_embeddings\":\(invalid)}".utf8)
        #expect(throws: SwevError.self) { try ModelConfigurationInspection(data: data, contextLimit: 1) }
    }
}

@Test func splitProcessorConfigurationPreservesDeclaredImageTokens() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    func write(_ file: String, _ json: String) throws {
        try Data(json.utf8).write(to: directory.appendingPathComponent(file))
    }
    try write("preprocessor_config.json", #"{"processor_class":"Idefics3Processor","max_image_size":{"longest_edge":384},"size":{"longest_edge":1536}}"#)
    try write("processor_config.json", #"{"image_seq_len":81,"size":{"longest_edge":999}}"#)
    try write("config.json", #"{"image_seq_len":81}"#)
    let data = try #require(try ProcessorCompatibility.configuration(directory: directory))
    let config = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(config["image_seq_len"] as? Int == 81)
    #expect((config["size"] as? [String: Int])?["longest_edge"] == 1536)
    try write("config.json", #"{"image_seq_len":64}"#)
    #expect(throws: SwevError.self) { try ProcessorCompatibility.configuration(directory: directory) }
    try write("processor_config.json", #"{}"#)
    try write("config.json", #"{}"#)
    #expect(throws: SwevError.self) { try ProcessorCompatibility.configuration(directory: directory) }
    try write("preprocessor_config.json", #"{"processor_class":"AnotherProcessor"}"#)
    #expect(try ProcessorCompatibility.configuration(directory: directory) == nil)
}

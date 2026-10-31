import XCTest
@testable import PulseLoop

/// Covers the Responses → Anthropic Messages translation. The network is stubbed with
/// `StubURLProtocol` (defined in CoachTests.swift); nothing here talks to the real API.
final class AnthropicClientTests: XCTestCase {
    private func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: config)
    }

    private func client(model: String = "claude-sonnet-5-5") -> AnthropicClient {
        AnthropicClient(apiKey: "sk-ant-test", model: model, session: session())
    }

    private func sentBody() throws -> [String: Any] {
        let data = try XCTUnwrap(StubURLProtocol.lastRequestBody)
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @discardableResult
    private func send(
        _ client: AnthropicClient,
        input: [[String: Any]] = [],
        tools: [[String: Any]] = [],
        previous: String? = nil,
        effort: String? = nil
    ) async throws -> OpenAIResponse {
        let body = try OpenAIRequestBuilder.data(
            model: "ignored", input: input, tools: tools, textFormat: nil,
            previousResponseId: previous, reasoningEffort: effort)
        return try await client.send(requestBody: body)
    }

    private func stub(_ json: String, status: Int = 200) {
        StubURLProtocol.statusCode = status
        StubURLProtocol.responseBody = Data(json.utf8)
    }

    private let okJSON = """
    {"id":"msg_1","type":"message","role":"assistant","stop_reason":"end_turn",\
    "content":[{"type":"text","text":"ok"}],\
    "usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}
    """

    func testEndpointAndHeaders() async throws {
        stub(okJSON)
        try await send(client(), input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])

        XCTAssertEqual(StubURLProtocol.lastRequestURL?.absoluteString, "https://api.anthropic.com/v1/messages")
        XCTAssertEqual(StubURLProtocol.lastRequestHeaders["x-api-key"], "sk-ant-test")
        XCTAssertEqual(StubURLProtocol.lastRequestHeaders["anthropic-version"], AnthropicClient.apiVersion)
        XCTAssertNil(StubURLProtocol.lastRequestHeaders["Authorization"])
    }

    func testMissingKeyThrows() async {
        let c = AnthropicClient(apiKey: "", session: session())
        do {
            try await send(c, input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])
            XCTFail("expected missingAPIKey")
        } catch ResponsesError.missingAPIKey {
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    /// The request must carry only what the Claude 5.5 family accepts: no `thinking`,
    /// `tool_choice` or sampling parameters (all are a 400 or deprecated), the system text
    /// at top level, and OpenAI-style tools converted to `input_schema`.
    func testRequestShape() async throws {
        stub(okJSON)
        let tools: [[String: Any]] = [
            ["type": "function", "name": "get_hr", "description": "Heart rate",
             "parameters": ["type": "object", "properties": [String: Any]()] as [String: Any]],
            ["type": "web_search"],
        ]
        try await send(
            client(),
            input: [
                OpenAIRequestBuilder.message(role: "developer", content: "You are a coach."),
                OpenAIRequestBuilder.message(role: "user", content: "hi"),
            ],
            tools: tools,
            effort: "high")

        let json = try sentBody()
        XCTAssertEqual(json["model"] as? String, "claude-sonnet-5-5")
        XCTAssertEqual(json["max_tokens"] as? Int, AnthropicClient.maxOutputTokens)
        XCTAssertTrue((json["system"] as? String)?.contains("You are a coach.") == true)
        for forbidden in ["thinking", "tool_choice", "temperature", "top_p", "top_k"] {
            XCTAssertNil(json[forbidden], "\(forbidden) must not be sent")
        }
        XCTAssertEqual((json["output_config"] as? [String: Any])?["effort"] as? String, "high")
        XCTAssertEqual((json["cache_control"] as? [String: Any])?["type"] as? String, "ephemeral")

        let sentTools = try XCTUnwrap(json["tools"] as? [[String: Any]])
        XCTAssertEqual(sentTools.count, 1, "hosted web_search has no equivalent and must be dropped")
        XCTAssertEqual(sentTools[0]["name"] as? String, "get_hr")
        XCTAssertNotNil(sentTools[0]["input_schema"])

        let messages = try XCTUnwrap(json["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"] as? String, "user")
        let blocks = try XCTUnwrap(messages[0]["content"] as? [[String: Any]])
        XCTAssertEqual(blocks.first?["text"] as? String, "hi")
    }

    /// An effort level the API doesn't define is dropped rather than sent (it would 400).
    func testUnknownEffortIsNotSent() async throws {
        stub(okJSON)
        try await send(client(), input: [OpenAIRequestBuilder.message(role: "user", content: "hi")], effort: "none")
        XCTAssertNil(try sentBody()["output_config"])
    }

    func testConsecutiveSameRoleMessagesMerge() async throws {
        stub(okJSON)
        try await send(client(), input: [
            OpenAIRequestBuilder.message(role: "user", content: "one"),
            OpenAIRequestBuilder.message(role: "user", content: "two"),
        ])
        let messages = try XCTUnwrap(try sentBody()["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual((messages[0]["content"] as? [[String: Any]])?.count, 2)
    }

    func testResponseParsingAndUsage() async throws {
        stub("""
        {"id":"msg_2","type":"message","role":"assistant","stop_reason":"tool_use",\
        "content":[{"type":"text","text":"Checking."},\
        {"type":"tool_use","id":"toolu_1","name":"get_hr","input":{"days":7}}],\
        "usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}
        """)
        let response = try await send(client(), input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])

        XCTAssertEqual(response.id, "msg_2")
        XCTAssertEqual(response.outputText, "Checking.")
        let call = try XCTUnwrap(response.functionCalls.first)
        XCTAssertEqual(call.name, "get_hr")
        XCTAssertEqual(call.callID, "toolu_1")
        let args = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(call.arguments.utf8)) as? [String: Any])
        XCTAssertEqual(args["days"] as? Int, 7)

        // Total input = uncached + cache read + cache write; cached is the read portion only.
        XCTAssertEqual(response.usage, CoachTokenUsage(inputTokens: 130, outputTokens: 5, cachedInputTokens: 100))
    }

    /// Preserved thinking validates the signature chain, so the assistant turn (thinking block
    /// included) must be replayed byte-for-byte, with the tool results in the next user message
    /// and ahead of any plain-text follow-up.
    func testContinuationReplaysAssistantContentVerbatim() async throws {
        let c = client()
        stub("""
        {"id":"msg_a","type":"message","role":"assistant","stop_reason":"tool_use",\
        "content":[{"type":"thinking","thinking":"","signature":"sig/abc=="},\
        {"type":"tool_use","id":"toolu_1","name":"get_hr","input":{}}],\
        "usage":{"input_tokens":1,"output_tokens":1}}
        """)
        try await send(c, input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])

        stub(okJSON)
        try await send(c, input: [
            OpenAIRequestBuilder.functionCallOutput(callID: "toolu_1", output: "{\"hr\":60}"),
            OpenAIRequestBuilder.message(role: "user", content: "Reply as valid JSON."),
        ], previous: "msg_a")

        let messages = try XCTUnwrap(try sentBody()["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.map { $0["role"] as? String }, ["user", "assistant", "user"])

        let assistant = try XCTUnwrap(messages[1]["content"] as? [[String: Any]])
        XCTAssertEqual(assistant[0]["type"] as? String, "thinking")
        XCTAssertEqual(assistant[0]["signature"] as? String, "sig/abc==")
        XCTAssertEqual(assistant[1]["type"] as? String, "tool_use")

        let user = try XCTUnwrap(messages[2]["content"] as? [[String: Any]])
        XCTAssertEqual(user[0]["type"] as? String, "tool_result")
        XCTAssertEqual(user[0]["tool_use_id"] as? String, "toolu_1")
        XCTAssertEqual(user[1]["type"] as? String, "text")
    }

    func testRefusalSurfacesAsError() async {
        stub("""
        {"id":"msg_r","type":"message","role":"assistant","stop_reason":"refusal",\
        "stop_details":{"type":"refusal","category":"cyber"},"content":[],\
        "usage":{"input_tokens":1,"output_tokens":0}}
        """)
        do {
            try await send(client(), input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])
            XCTFail("expected a refusal error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("declined"))
        }
    }

    func testHTTPErrorIsReported() async {
        stub(#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#, status: 401)
        do {
            try await send(client(), input: [OpenAIRequestBuilder.message(role: "user", content: "hi")])
            XCTFail("expected an HTTP error")
        } catch ResponsesError.http(let status, _) {
            XCTAssertEqual(status, 401)
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }
}

import Foundation

/// Adapts the app's `ResponsesClient` protocol to Anthropic's Messages API
/// (`POST /v1/messages`). Translates OpenAI Responses-style request bodies into
/// Messages requests and maps the result back, so the orchestrator never needs to
/// know which provider is active. There is no official Anthropic Swift SDK, so
/// this is a thin `URLSession` client, like the other providers.
///
/// State across turns: the Responses API is stateful (`previous_response_id`);
/// the Messages API is stateless. This class accumulates the conversation as
/// Messages `messages` across `send` calls. The orchestrator creates a fresh
/// client per agent turn via the factory, so one instance covers one turn.
///
/// Model-specific rules this client relies on (Claude 5.5 family):
///  - `thinking` is deliberately never sent. Omitting it runs adaptive thinking;
///    `{type: "disabled"}` / `budget_tokens` are rejected with a 400.
///  - `tool_choice` is never sent (forced `any`/`tool` is a 400; `auto` is default).
///  - Sampling parameters (`temperature`, `top_p`, `top_k`) are never sent.
///  - Assistant `content` blocks (including `thinking`) are replayed *verbatim*
///    between tool rounds. Preserved thinking validates the signature chain, so
///    editing or dropping a block from the history can invalidate the request.
///  - No assistant prefill (a trailing assistant message is a 400).
///
/// Not supported in this first version: hosted web search (the `web_search`
/// spec is dropped, and Settings hides the toggle) and server-enforced JSON
/// output (`output_config.format`). The coach's output shape is carried by
/// `CoachResponseSchema.promptInstruction` plus the orchestrator's JSON-repair
/// loop, exactly as for MiniMax and the local provider.
final class AnthropicClient: ResponsesClient, @unchecked Sendable {
    static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    static let apiVersion = "2026-10-21"
    /// Thinking tokens count toward this, so leave room beyond the visible JSON.
    static let maxOutputTokens = 16000
    /// Generous: adaptive thinking on Opus can legitimately run for a while.
    static let requestTimeout: TimeInterval = 180

    private static let validEfforts: Set<String> = ["low", "medium", "high", "xhigh", "max"]

    private let apiKey: String
    private let model: String
    private let session: URLSession

    private var systemText: String = ""
    private var messages: [[String: Any]] = []
    /// Response id → the assistant `content` array exactly as the API returned it
    /// (thinking + text + tool_use blocks), replayed before the matching tool results.
    private var storedAssistantContent: [String: [[String: Any]]] = [:]

    init(apiKey: String, model: String = AnthropicModel.default.rawValue, session: URLSession = .shared) {
        self.apiKey = apiKey
        self.model = model
        self.session = session
    }

    func send(requestBody: Data) async throws -> OpenAIResponse {
        guard !apiKey.isEmpty else { throw ResponsesError.missingAPIKey }

        guard let req = try? JSONSerialization.jsonObject(with: requestBody) as? [String: Any] else {
            throw ResponsesError.decoding("AnthropicClient: invalid request body")
        }

        let input = req["input"] as? [[String: Any]] ?? []
        let tools = req["tools"] as? [[String: Any]] ?? []
        let previousResponseId = req["previous_response_id"] as? String
        let effort = ((req["reasoning"] as? [String: Any])?["effort"] as? String)?.lowercased()

        if let previousResponseId {
            appendContinuation(previousId: previousResponseId, input: input)
        } else {
            setupConversation(from: input)
        }

        let body = buildBody(tools: convertTools(tools), effort: effort)
        let bodyData = try JSONSerialization.data(withJSONObject: body, options: [.withoutEscapingSlashes])

        var request = URLRequest(url: Self.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue(Self.apiVersion, forHTTPHeaderField: "anthropic-version")
        request.httpBody = bodyData
        request.timeoutInterval = Self.requestTimeout

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw ResponsesError.transport(error)
        }

        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw ResponsesError.http(status: http.statusCode, body: body)
        }

        return try parseMessagesResponse(data)
    }

    // MARK: - Conversation setup

    /// First turn: lift system/developer items into the top-level `system` string
    /// and convert the rest into alternating user/assistant messages.
    private func setupConversation(from input: [[String: Any]]) {
        messages = []
        storedAssistantContent = [:]

        var systemParts: [String] = []
        for item in input {
            guard let role = item["role"] as? String, item["content"] != nil else { continue }
            if role == "system" || role == "developer" {
                if let text = item["content"] as? String, !text.isEmpty { systemParts.append(text) }
                continue
            }
            let blocks = contentBlocks(from: item)
            guard !blocks.isEmpty else { continue }
            appendMessage(role: role == "assistant" ? "assistant" : "user", blocks: blocks)
        }
        // The API requires the first message to be a user turn.
        while messages.first?["role"] as? String == "assistant" { messages.removeFirst() }

        // No enforced `output_config.format`, so tell the model the output shape out-of-band.
        systemParts.append(CoachResponseSchema.promptInstruction)
        systemText = systemParts.joined(separator: "\n\n")
    }

    /// Later turns: replay the stored assistant content for `previousId`, then one
    /// user message carrying every tool result (first, as the API requires) followed
    /// by any plain-text repair/follow-up messages.
    private func appendContinuation(previousId: String, input: [[String: Any]]) {
        if let content = storedAssistantContent[previousId] {
            appendMessage(role: "assistant", blocks: content)
        }

        var toolResults: [[String: Any]] = []
        var followUps: [[String: Any]] = []
        for item in input {
            if (item["type"] as? String) == "function_call_output",
               let callId = item["call_id"] as? String,
               let output = item["output"] as? String {
                var block: [String: Any] = ["type": "tool_result", "tool_use_id": callId]
                // An empty content string is rejected; give the model something explicit.
                block["content"] = output.isEmpty ? "(empty result)" : output
                toolResults.append(block)
            } else if let role = item["role"] as? String, item["content"] != nil {
                if role == "system" || role == "developer" {
                    // Mid-conversation system messages are model-dependent; fold into the user turn.
                    if let text = item["content"] as? String, !text.isEmpty {
                        followUps.append(["type": "text", "text": text])
                    }
                } else {
                    followUps.append(contentsOf: contentBlocks(from: item))
                }
            }
        }
        let userBlocks = toolResults + followUps
        if !userBlocks.isEmpty { appendMessage(role: "user", blocks: userBlocks) }
    }

    /// Appends a message, merging into the previous one when the role repeats so
    /// the conversation always alternates.
    private func appendMessage(role: String, blocks: [[String: Any]]) {
        if var last = messages.last, last["role"] as? String == role,
           var existing = last["content"] as? [[String: Any]] {
            existing.append(contentsOf: blocks)
            last["content"] = existing
            messages[messages.count - 1] = last
        } else {
            messages.append(["role": role, "content": blocks])
        }
    }

    // MARK: - Content conversion

    /// Converts a Responses message item's `content` into Messages content blocks.
    /// Plain strings become one text block; the image-bearing content-part array
    /// (`input_text` + `input_image`) becomes text + base64 image blocks. Empty text
    /// is dropped because the API rejects empty text blocks.
    private func contentBlocks(from item: [String: Any]) -> [[String: Any]] {
        if let text = item["content"] as? String {
            return text.isEmpty ? [] : [["type": "text", "text": text]]
        }
        guard let parts = item["content"] as? [[String: Any]] else { return [] }
        var out: [[String: Any]] = []
        for part in parts {
            switch part["type"] as? String {
            case "input_text", "text", "output_text":
                if let text = part["text"] as? String, !text.isEmpty { out.append(["type": "text", "text": text]) }
            case "input_image":
                if let source = imageSource(fromDataURL: part["image_url"] as? String) {
                    out.append(["type": "image", "source": source])
                }
            default:
                break
            }
        }
        return out
    }

    /// Splits `data:<mime>;base64,<data>` into the Messages `base64` image source.
    private func imageSource(fromDataURL url: String?) -> [String: String]? {
        guard let url, url.hasPrefix("data:"),
              let comma = url.firstIndex(of: ","),
              let semicolon = url.firstIndex(of: ";"),
              semicolon < comma
        else { return nil }
        let mime = String(url[url.index(url.startIndex, offsetBy: 5)..<semicolon])
        let data = String(url[url.index(after: comma)...])
        guard !mime.isEmpty, !data.isEmpty else { return nil }
        return ["type": "base64", "media_type": mime, "data": data]
    }

    // MARK: - Tool conversion (Responses flat → Messages)

    /// `{type: function, name, description, parameters}` → `{name, description,
    /// input_schema}`. The OpenAI-hosted `web_search` spec is dropped.
    private func convertTools(_ tools: [[String: Any]]) -> [[String: Any]] {
        tools.compactMap { tool -> [String: Any]? in
            let type = tool["type"] as? String
            if type == "web_search" || type == "web_search_preview" { return nil }
            guard type == "function", let name = tool["name"] as? String else { return nil }
            var out: [String: Any] = ["name": name]
            if let desc = tool["description"] as? String { out["description"] = desc }
            out["input_schema"] = (tool["parameters"] as? [String: Any])
                ?? ["type": "object", "properties": [String: Any]()]
            return out
        }
    }

    // MARK: - Request body

    private func buildBody(tools: [[String: Any]], effort: String?) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "max_tokens": Self.maxOutputTokens,
            "messages": messages,
            // Automatic prompt caching of the stable prefix (tools → system → messages).
            // The coach resends a large context packet every round, so this matters for cost.
            "cache_control": ["type": "ephemeral"],
        ]
        if !systemText.isEmpty { body["system"] = systemText }
        if !tools.isEmpty { body["tools"] = tools }
        if let effort, Self.validEfforts.contains(effort) { body["output_config"] = ["effort": effort] }
        return body
    }

    // MARK: - Response parsing

    private func parseMessagesResponse(_ data: Data) throws -> OpenAIResponse {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ResponsesError.decoding("AnthropicClient: response was not a JSON object")
        }
        if (root["type"] as? String) == "error", let err = root["error"] as? [String: Any] {
            throw ResponsesError.decoding("Anthropic error: \(err["message"] as? String ?? "unknown error")")
        }

        let stopReason = root["stop_reason"] as? String
        if stopReason == "refusal" {
            let category = (root["stop_details"] as? [String: Any])?["category"] as? String
            throw ResponsesError.decoding(
                "Claude declined this request" + (category.map { " (\($0))" } ?? "") + ".")
        }

        guard let content = root["content"] as? [[String: Any]] else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw ResponsesError.decoding("AnthropicClient: no content in response — \(body.prefix(300))")
        }

        var outputItems: [ResponseOutputItem] = []
        for block in content {
            switch block["type"] as? String {
            case "text":
                if let text = block["text"] as? String, !text.isEmpty { outputItems.append(.message(text: text)) }
            case "tool_use":
                guard let name = block["name"] as? String, let id = block["id"] as? String else { continue }
                let input = block["input"] as? [String: Any] ?? [:]
                let args = (try? JSONSerialization.data(withJSONObject: input, options: [.withoutEscapingSlashes]))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                outputItems.append(.functionCall(ResponseFunctionCall(name: name, callID: id, arguments: args)))
            default:
                // thinking / redacted_thinking: no visible output, but kept in `content` for replay.
                break
            }
        }

        // A cut-off tool call carries partial arguments; surface it instead of running it.
        if stopReason == "max_tokens", outputItems.contains(where: { if case .functionCall = $0 { return true } else { return false } }) {
            throw ResponsesError.decoding("Claude's response was cut off mid tool call (max_tokens).")
        }
        if outputItems.isEmpty { throw ResponsesError.emptyOutput }

        let responseId = (root["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? UUID().uuidString
        storedAssistantContent[responseId] = content
        return OpenAIResponse(id: responseId, outputItems: outputItems, usage: usage(from: root))
    }

    /// Messages reports uncached input, cache reads, and cache writes separately.
    /// `CoachTokenUsage.inputTokens` is the *total* input with cached tokens as a
    /// subset, so sum all three and report reads as the cached portion.
    private func usage(from root: [String: Any]) -> CoachTokenUsage? {
        guard let usage = root["usage"] as? [String: Any] else { return nil }
        let uncached = usage["input_tokens"] as? Int ?? 0
        let cacheRead = usage["cache_read_input_tokens"] as? Int ?? 0
        let cacheWrite = usage["cache_creation_input_tokens"] as? Int ?? 0
        return CoachTokenUsage(
            inputTokens: uncached + cacheRead + cacheWrite,
            outputTokens: usage["output_tokens"] as? Int ?? 0,
            cachedInputTokens: cacheRead
        )
    }
}

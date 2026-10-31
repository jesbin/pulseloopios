import Foundation

/// Single source of truth for "which `ResponsesClient` runs, given the user's
/// settings + stored keys." Shared by the chat view-model, the summary service,
/// and the notification service so provider logic lives in exactly one place.
///
/// The returned `key` is a readiness sentinel: non-`nil` means the provider can
/// run (used to build `CoachFeatureFlags.hasAPIKey`). For cloud providers it's
/// the actual key; for on-device it's a `"on-device"` placeholder.
@MainActor
enum CoachClientResolver {
    static func resolve(
        settings: CoachSettings,
        openAIKeyStore: APIKeyStore,
        geminiKeyStore: APIKeyStore,
        openRouterKeyStore: APIKeyStore,
        minimaxKeyStore: APIKeyStore,
        // Defaulted so the four existing call sites don't each have to learn about a provider
        // whose key is optional anyway.
        localKeyStore: APIKeyStore = LocalLLMKeychainStore(),
        anthropicKeyStore: APIKeyStore = AnthropicKeychainStore(),
        openAIClientFactory: (String) -> ResponsesClient = { OpenAIResponsesClient(apiKey: $0) }
    ) -> (key: String?, client: ResponsesClient) {
        switch settings.providerMode {
        case .appleOnDevice:
            // On-device only — no cloud backup. When the local model is usable,
            // run it; otherwise hand back the client (it throws a clear error
            // that surfaces in chat) and signal "not ready" so generators degrade
            // to scripted.
            let onDevice = AppleFoundationModelsClient()
            let available = AppleOnDeviceAvailability.current.isAvailable
            return (available ? "on-device" : nil, onDevice)
        default:
            let stores = KeyStores(
                openAI: openAIKeyStore, gemini: geminiKeyStore, openRouter: openRouterKeyStore,
                minimax: minimaxKeyStore, local: localKeyStore, anthropic: anthropicKeyStore)
            return directClient(
                settings.providerMode, settings: settings, stores: stores,
                openAIClientFactory: openAIClientFactory)
        }
    }

    /// The per-provider key stores, bundled so `directClient` stays under SwiftLint's
    /// parameter-count limit as providers are added.
    private struct KeyStores {
        let openAI: APIKeyStore
        let gemini: APIKeyStore
        let openRouter: APIKeyStore
        let minimax: APIKeyStore
        let local: APIKeyStore
        let anthropic: APIKeyStore
    }

    /// Builds a client for a concrete (non-on-device) provider, mirroring the
    /// prior per-call-site logic. Returns a client even when the key is absent
    /// (`key == nil`); the feature-flags gate prevents an empty-key call.
    private static func directClient(
        _ mode: CoachProviderMode,
        settings: CoachSettings,
        stores: KeyStores,
        openAIClientFactory: (String) -> ResponsesClient
    ) -> (key: String?, client: ResponsesClient) {
        switch mode {
        case .userAnthropicKey:
            let key = (try? stores.anthropic.readKey()) ?? nil
            return (key, AnthropicClient(apiKey: key ?? "", model: settings.anthropicModel))
        case .userGeminiKey:
            let key = (try? stores.gemini.readKey()) ?? nil
            return (key, GeminiClient(apiKey: key ?? ""))
        case .userOpenRouterKey:
            let key = (try? stores.openRouter.readKey()) ?? nil
            return (key, OpenRouterClient(
                apiKey: key ?? "",
                model: settings.openRouterModel,
                privacyRouting: settings.orEnablePrivacyRouting,
                providerSort: settings.orProviderSort))
        case .userMiniMaxKey:
            let key = (try? stores.minimax.readKey()) ?? nil
            return (key, MiniMaxClient(apiKey: key ?? "", model: settings.minimaxModel))
        case .localOpenAICompat:
            // Readiness is a base URL that would actually work — `validate`, not "non-empty".
            // Settings persists the field as the user types, so a non-empty check would flip the
            // coach to "Active" on the first character and every turn would then fail inside
            // `send()` with the same URL error the Settings field is already showing inline.
            // The key may legitimately be absent and is passed through as nil so the client omits
            // the Authorization header entirely.
            let baseURL = settings.resolvedLocalBaseURL
            let key = (try? stores.local.readKey()) ?? nil
            let ready = LocalEndpoint.validate(baseURL) == nil ? baseURL : nil
            return (ready, LocalOpenAICompatClient(
                baseURL: baseURL,
                model: settings.resolvedLocalModel,
                apiKey: key,
                toolCallingEnabled: settings.localToolCalling,
                structuredOutput: settings.localStructuredOutput,
                maxOutputTokens: settings.localMaxTokens > 0 ? settings.localMaxTokens : nil,
                readTimeoutSeconds: settings.localTimeoutSeconds))
        default:
            // userOpenAIKey / offlineStub / backendProxy (and appleOnDevice never
            // reaches here) all use the OpenAI key + factory.
            let key = (try? stores.openAI.readKey()) ?? nil
            return (key, openAIClientFactory(key ?? ""))
        }
    }
}

//
//  LlamaModel.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/11/25.
//

import Foundation
#if os(iOS)
import class UIKit.UIDevice
#endif
@MainActor
class LlamaState: ObservableObject {

    /// 生成中であるかどうか
    @Published var isGenerating = false
    
    @Published var chatLog = ChatLog()

    private var generationTask: Task<Void, any Error>?

    struct ChatLog {
        enum Role: Sendable, Equatable, Hashable {
            case user, ai, system
        }
        struct Message: Sendable, Equatable, Hashable, Identifiable {
            var role: Role
            var message: String
            let id = UUID()
        }
        var messages: [Message] = []

        mutating func updateLastAiMessage(_ message: String) {
            self.messages[self.messages.endIndex - 1].message.removeAll()
            self.messages[self.messages.endIndex - 1].message.append(contentsOf: message)
        }
    }

    /// 生成中のテキスト
    private var generatingMessage: String = ""
    private var llamaContext: LlamaContext?
    private var modelUrl: URL? {
        #if os(macOS)
        Bundle.main.url(forResource: "ELYZA-japanese-Llama-2-7b-instruct-q4_K_M", withExtension: "gguf")
        #elseif os(iOS)
        if UIDevice.current.userInterfaceIdiom == .pad {
            Bundle.main.bundleURL.appending(path: "ELYZA-japanese-Llama-2-7b-instruct-q4_K_M.gguf")
        } else {
            Bundle.main.bundleURL.appending(path: "tinyllama-1.1b-intermediate-step-715k-1.5t.Q2_K.gguf")
        }
        #endif
    }
    init() {
        do {
            self.chatLog.messages.append(.init(role: .system, message: "Loading model..."))
            if let modelUrl {
                self.llamaContext = try LlamaContext.createContext(path: modelUrl.path())
                self.chatLog.messages.append(.init(role: .system, message: "Loaded model \(modelUrl.lastPathComponent)"))
            } else {
                self.chatLog.messages.append(.init(role: .system, message: "Could not find model of specified url"))
            }
        } catch {
            print(error)
        }
    }

    /// - parameters:
    ///   - prompt: text to give the model
    ///   - createNewContext: `true` if you want to clear the current context
    @MainActor
    func refreshContext() {
        self.chatLog.messages = []
        self.isGenerating = false
        self.generatingMessage = ""
        Task {
            try await self.llamaContext?.reset_context()
        }
    }

    /// - parameters:
    ///   - prompt: text to give the model
    ///   - createNewContext: `true` if you want to clear the current context
    @MainActor
    func addChatMessage(_ message: String, instruction: String, example: [ChatLog.Message], userMessagePrefix: String, aiMessagePrefix: String) {
        self.isGenerating = true
        self.generatingMessage = ""
        self.chatLog.messages.append(.init(role: .user, message: message))
        self.chatLog.messages.append(.init(role: .ai, message: ""))
        var keptMessages: [ChatLog.Message] = []
        var keptLength = 0
        for message in chatLog.messages.reversed() where message.role != .system {
            keptMessages.append(message)
            keptLength += message.message.utf8.count
            if message.role == .user && keptLength > 512 {
                break
            }
        }
        print("Request with prompt", keptMessages)
        self.generationTask = Task {
            await llamaContext?.clear()
            let prompt = instruction + (example + keptMessages.reversed()).compactMap {
                switch $0.role {
                case .system:
                    nil
                case .user:
                    userMessagePrefix + $0.message
                case .ai:
                    aiMessagePrefix + $0.message
                }
            }.joined(separator: "\n")
            print("Request with prompt", prompt.suffix(20))
            await self.generateForChat(prompt: prompt, userPromptPrefix: userMessagePrefix, aiPromptPrefix: aiMessagePrefix)
        }
    }

    @MainActor
    func stopGenerationTask() {
        guard self.isGenerating else {
            return
        }
        self.generatingMessage = ""
        self.isGenerating = false
        self.generationTask?.cancel()
    }

    enum GenerationError: Error {
        case interrupt
        case noLlamaContext
    }

    /// - parameters:
    ///   - prompt: 与えるプロンプト
    ///   - length: 生成する長さ
    ///   - reversePrompt: ユーザ側の入力を待つ条件
    func generateForChat(prompt: String, userPromptPrefix: String?, aiPromptPrefix: String?) async {
        guard let llamaContext else {
            return
        }
        guard let last = self.chatLog.messages.last, last.role == .ai else {
            return
        }
        var targetMessageId = last.id
        await llamaContext.completion_init(text: prompt)
        var reversed = false
        while await llamaContext.n_cur < llamaContext.n_len && !Task.isCancelled && !reversed {

            let completion = await llamaContext.completion_loop()
            var newResult = self.generatingMessage + completion.piece
            // reverse promptが発見されたら停止する
            if let userPromptPrefix, newResult.suffix(userPromptPrefix.count + 10).contains(userPromptPrefix) {
                while !newResult.hasSuffix(userPromptPrefix) {
                    newResult.removeLast()
                }
                newResult.removeLast(userPromptPrefix.count)
                if newResult.hasSuffix("\n") {
                    newResult.removeLast()
                }
                // ユーザ側に制御を戻す
                reversed = true
            }
            if completion.state == .eos {
                reversed = true
            }
            //
            if let aiPromptPrefix, newResult.contains(aiPromptPrefix) {
                let parts = newResult.split(separator: aiPromptPrefix, omittingEmptySubsequences: false)
                for (i, part) in zip(parts.indices, parts) {
                    var part = part
                    if part.hasSuffix("\n") {
                        part.removeLast()
                    }
                    self.chatLog.updateLastAiMessage(String(part))
                    if i == parts.endIndex - 1 {
                        self.generatingMessage = String(part)
                    } else{
                        let newMessage = ChatLog.Message(role: .ai, message: "")
                        targetMessageId = newMessage.id
                        self.chatLog.messages.append(newMessage)
                    }
                }
            } else {
                // 更新する
                do {
                    try await MainActor.run {
                        // interruptがないか確認する
                        guard let last = self.chatLog.messages.last, last.id == targetMessageId else {
                            throw GenerationError.interrupt
                        }
                        self.generatingMessage = newResult
                        self.chatLog.updateLastAiMessage(newResult)
                    }
                } catch {
                    break
                }
            }
        }
        print("Done")
        if !Task.isCancelled {
            await llamaContext.clear()
            self.isGenerating = false
        }
    }

    /// - parameters:
    ///   - prompt: 与えるプロンプト
    ///   - length: 生成する長さ
    ///   - reversePrompt: ユーザ側の入力を待つ条件
    func generateWithGrammar(prompt: String, grammar: LlamaGrammar) async throws -> String {
        guard let llamaContext else {
            throw GenerationError.noLlamaContext
        }
        await llamaContext.completion_init(text: prompt)
        var result = ""
        while await llamaContext.n_cur < llamaContext.n_len && !Task.isCancelled {
            let completion = await llamaContext.completion_loop_with_grammar(grammar: grammar)
            result.append(contentsOf: completion.piece)
            if result.contains(#/\n+/#) {
                break
            }
            if completion.state != .normal {
                break
            }
        }
        if !Task.isCancelled {
            await llamaContext.clear()
            self.isGenerating = false
        }
        print("Done")
        return result
    }
}

//
//  ContentView.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/11/25.
//

import SwiftUI

struct ContentView: View {
    private var instruction: String {
//        "Transcript of a dialog, where the User interacts with an AI Assistant named Alan. Alan is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision."
        "以下は会話の書き起こしで、ユーザはAlanというAIアシスタントと会話しています。Alanは優しく、正直で、役立ち、文章がうまく、決してユーザのリクエストに応えることに失敗せず、常に正確です。"
    }
    private var chatExample: [LlamaState.ChatLog.Message] {
        [
            .init(role: .user, message: "やあAlan"),
            .init(role: .ai, message: "こんにちは、何かお手伝いできることはありますか？"),
            .init(role: .user, message: "ヨーロッパで最も大きい都市はどこ？"),
            .init(role: .ai, message: "モスクワです。"),
            .init(role: .ai, message: "ロシアの首都です。"),
        ]
    }
    @State private var message: String = "東京の観光スポットを10個教えて"
    @StateObject private var model = LlamaState()

    var body: some View {
        VStack {
            TextField("Insert prompt", text: $message)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button {
                    self.model.addChatMessage(message, instruction: instruction, example: chatExample, userMessagePrefix: "User:", aiMessagePrefix: "Alan:")
                    self.message = ""
                } label: {
                    Label {
                        Text("Generate")
                    } icon: {
                        Image(systemName: "circle.hexagongrid.fill")
                            .foregroundStyle(.conicGradient(AnyGradient(Gradient(colors: [.yellow, .blue])), angle: Angle(degrees: 90)))
                    }
                }
                .keyboardShortcut(.return, modifiers: .command)
                if self.model.isGenerating {
                    Button("Cancel", systemImage: "stop.circle") {
                        self.model.stopGenerationTask()
                    }
                } else {
                    Button("Reset", systemImage: "repeat") {
                        self.model.refreshContext()
                    }
                }
            }
            ScrollView {
                ScrollViewReader { proxy in
                    VStack(alignment: .leading) {
                        ForEach(self.model.chatLog.messages) { message in
                            HStack(alignment: .top) {
                                let (systemImage, backgroundCornerRadii, backgroundColor): (String, RectangleCornerRadii, Color) = switch message.role {
                                case .system:
                                    ("info.bubble", .init(topLeading: 0, bottomLeading: 10, bottomTrailing: 10, topTrailing: 10), .gray)
                                case .ai:
                                    ("poweroutlet.type.b", .init(topLeading: 0, bottomLeading: 10, bottomTrailing: 10, topTrailing: 10), .orange)
                                case .user:
                                    ("person.fill", .init(topLeading: 0, bottomLeading: 10, bottomTrailing: 10, topTrailing: 10), .green)
                                }
                                Image(systemName: systemImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(width: 25)
                                Text(message.message)
                                    .textSelection(.enabled)
                                    .padding()
                                    .background {
                                        UnevenRoundedRectangle(cornerRadii: backgroundCornerRadii)
                                            .fill(backgroundColor)
                                    }
                            }
                            .id(message.id)
                        }
                    }
                    .onChange(of: self.model.chatLog.messages.count) { _ in
                        proxy.scrollTo(self.model.chatLog.messages.last?.id, anchor: .bottom)
                    }
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

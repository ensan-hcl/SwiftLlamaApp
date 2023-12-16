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
                            HStack {
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
        .task {
            print(LlamaGrammar.japanese)
            guard let jsonGrammar = LlamaGrammar.japanese else {
                return
            }
            let result = try? await self.model.generateWithGrammar(prompt: "私の名前は", grammar: jsonGrammar)

//            self.model.chatLog.messages.append(contentsOf: [
//                .init(role: .user, message: "沖縄の観光スポットを10個教えて"),
//                .init(role: .ai, message: " 沖縄は観光スポットが多くて選べませんが、沖縄美ら海水族館、北谷ビーチ、沖縄戦跡、首里城、沖縄モノレール、沖縄セルラーパーク、桜島、牧志公園、恩納村、浜比嘉島が人気です。"),
//                .init(role: .user, message: "東京の観光スポットを10個教えて"),
//                .init(role: .ai, message: " 東京の観光スポットは多くて選べませんが、東京タワー、東京駅、浅草寺、上野公園、増上寺、国立西洋美術館、新宿御苑、明治神宮、国立科学博物館、日本民芸館、浅草橋が人気です。"),
//                .init(role: .user, message: "北海道の観光スポットを10個教えて"),
//                .init(role: .ai, message: " 北海道の観光スポットは多くて選べませんが、ニセコ、富良野、十勝、札幌、小樽、函館、倶知安、洞爺湖、美瑛、積丹岬が人気です。"),
//                .init(role: .user, message: "大阪の観光スポットを10個教えて"),
//                .init(role: .ai, message: " 大阪の観光スポットは多くて選べませんが、USJ、大阪城、天保山、グリーンヒルホテル大阪、通天閣、グローバルランド、海遊館、大阪ミニatureミュージアム、大阪市科学館、大阪ファミリーパーク、アサヒビール記念館が人気です。")
//            ])
        }
    }
}

#Preview {
    ContentView()
}

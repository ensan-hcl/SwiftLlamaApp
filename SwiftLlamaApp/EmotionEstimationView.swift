//
//  EmotionEstimationView.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/12/17.
//

import SwiftUI

private struct Emotion: Codable {
    var sadness: Int
    var joy: Int
    var anger: Int
}

struct EmotionEstimationView: View {
    @State private var reviewText: String = ""
    @StateObject private var model = LlamaState()
    @State private var emotion: Emotion?
    @State private var inProgress = false
    var body: some View {
        TextField("レビュー", text: $reviewText)
        Button("感情推定！") {
            Task {
                self.inProgress = true
                self.emotion = await getEmotion(reviewText: reviewText)
                self.inProgress = false
            }
        }
        if let emotion {
            HStack {
                Text("😆: \(emotion.joy)")
                Text("😡: \(emotion.anger)")
                Text("😭: \(emotion.sadness)")
            }
        } else if inProgress {
            ProgressView()
        }
    }

    private func getEmotion(reviewText: String) async -> Emotion? {
        guard let jsonGrammar = LlamaGrammar.json else {
            return nil
        }
        guard let result = try? await self.model.generateWithGrammar(prompt: """
        The following data is emotion estimation data of user review. There are three metris including "sadness", "joy", and "anger". For each key, strength value of 0-5 is applied.

        Review: "This is a great app!"
        {"joy": 5, "anger": 0, "sadness": 0}

        Review: "\(reviewText)"
        """, grammar: jsonGrammar) else {
            return nil
        }
        return try? JSONDecoder().decode(Emotion.self, from: result.data(using: .utf8)!)
    }
}

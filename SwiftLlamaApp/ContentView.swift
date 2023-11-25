//
//  ContentView.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/11/25.
//

import SwiftUI

class ViewModel: ObservableObject {
    var llama: Llama?
    init() {
        do {
            #if os(macOS)
            let path = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/llama-2-13b.Q4_0.gguf", isDirectory: false).path()
            #elseif os(iOS)
            let path = Bundle.main.bundleURL.appendingPathComponent("llama-2-13b.Q4_0.gguf", isDirectory: false).path()
            #endif
            self.llama = try Llama(modelPath: path)
        } catch {
            print(error)
        }
    }

    /// 生成結果
    @Published var result: String = ""

    func generate(prompt: String, length: Int = 100) async {
        guard let stream = self.llama?.generate(prompt: prompt, n_len: length) else {
            await MainActor.run {
                result = "llama model could not be loaded"
            }
            return
        }
        await MainActor.run {
            result = prompt
        }
        for await value in stream where !Task.isCancelled {
            await MainActor.run {
                self.result.append(contentsOf: value)
            }
        }
    }
}

struct ContentView: View {
    @State private var generationTask: Task<Void, any Error>?
    @State private var prompt: String = "Let's learn Swift. "
    @StateObject private var model = ViewModel()
    var body: some View {
        VStack {
            TextField("Insert prompt", text: $prompt)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button {
                    self.generationTask?.cancel()
                    self.generationTask = Task {
                        await self.model.generate(prompt: prompt)
                    }
                } label: {
                    Label {
                        Text("Generate")
                    } icon: {
                        Image(systemName: "circle.hexagongrid.fill")
                            .foregroundStyle(.conicGradient(AnyGradient(Gradient(colors: [.yellow, .blue])), angle: Angle(degrees: 90)))
                    }
                }
                if generationTask?.isCancelled == false {
                    Button("Cancel", systemImage: "stop.circle") {
                        self.generationTask?.cancel()
                        self.generationTask = nil
                    }
                }
            }

            Text(verbatim: model.result)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

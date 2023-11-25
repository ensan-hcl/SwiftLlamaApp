//
//  LlamaModel.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/11/25.
//
//  This code is based on https://github.com/ggerganov/llama.cpp/tree/master/examples/batched.swift . The code is under MIT LICENSE.

import Foundation
import llama

class Llama {
    var model: Model
    enum LlamaError: Error {
        case modelLoadError
    }
    class Model {
        var _model: OpaquePointer
        init(modelPath: String) throws {
            let model_params = llama_model_default_params()
            guard let model = llama_load_model_from_file(modelPath.cString(using: .utf8), model_params) else {
                print("Failed to load model")
                throw LlamaError.modelLoadError
            }
            self._model = model
        }
        deinit {
            llama_free_model(_model)
        }
    }
    init(modelPath: String) throws {
        llama_backend_init(false)
        self.model = try Model(modelPath: modelPath)
    }
    deinit {
        llama_backend_free()
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let n_tokens = text.count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(model._model, text, Int32(text.count), tokens, Int32(n_tokens), add_bos, /*special tokens*/ false)
        var swiftTokens: [llama_token] = []
        for i in 0 ..< tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }
        tokens.deallocate()
        return swiftTokens
    }

    private func token_to_piece(token: llama_token, buffer: inout [CChar]) -> String? {
        var result = [CChar](repeating: 0, count: 8)
        let nTokens = llama_token_to_piece(model._model, token, &result, Int32(result.count))
        if nTokens < 0 {
            if result.count >= -Int(nTokens) {
                result.removeLast(-Int(nTokens))
            } else {
                result.removeAll()
            }
            let check = llama_token_to_piece(
                model._model,
                token,
                &result,
                Int32(result.count)
            )
            assert(check == nTokens)
        } else {
            result.removeLast(result.count - Int(nTokens))
        }
        if buffer.isEmpty, let utfString = String(cString: result + [0], encoding: .utf8) {
            return utfString
        } else {
            buffer.append(contentsOf: result)
            let data = Data(buffer.map { UInt8(bitPattern: $0) })
            if buffer.count >= 4 { // 4 bytes is the max length of a utf8 character so if we're here we need to reset the buffer
                buffer = []
            }
            guard let bufferString = String(data: data, encoding: .utf8) else {
                return nil
            }
            buffer = []
            return bufferString
        }
    }

    func generate(prompt: String, n_len: Int) -> AsyncStream<String> {
        AsyncStream<String> { continuation in
            DispatchQueue.global().async {
                let tokens = self.tokenize(text: prompt, add_bos: true)
                let n_kv_req = UInt32(tokens.count) + UInt32((n_len - Int(tokens.count)))
                var context_params = llama_context_default_params()
                context_params.seed = 1
                context_params.n_ctx = n_kv_req
                context_params.n_batch = UInt32(max(n_len, 1))
                context_params.n_threads = 8
                context_params.n_threads_batch = 8

                let context = llama_new_context_with_model(self.model._model, context_params)
                guard context != nil else {
                    print("Failed to initialize context")
                    exit(1)
                }

                defer {
                    llama_free(context)
                }

                let n_ctx = llama_n_ctx(context)

                print("\nn_len = \(n_len), n_ctx = \(n_ctx), n_batch = \(context_params.n_batch), n_kv_req = \(n_kv_req)\n")

                if n_kv_req > n_ctx {
                    print("error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", n_kv_req)
                    exit(1)
                }

                var buffer: [CChar] = []
                for id: llama_token in tokens {
                    print(self.token_to_piece(token: id, buffer: &buffer) ?? "", terminator: "")
                }

                print("\n")

                var batch = llama_batch_init(max(Int32(tokens.count), 1), 0, 1)
                defer {
                    llama_batch_free(batch)
                }

                // evaluate the initial prompt
                batch.n_tokens = Int32(tokens.count)

                for (i, token) in tokens.enumerated() {
                    batch.token[i] = token
                    batch.pos[i] = Int32(i)
                    batch.n_seq_id[i] = 1
                    // batch.seq_id[i][0] = 0
                    // TODO: is this the proper way to do this?
                    if let seq_id = batch.seq_id[i] {
                        seq_id[0] = 0
                    }
                    batch.logits[i] = 0
                }

                // llama_decode will output logits only for the last token of the prompt
                batch.logits[Int(batch.n_tokens) - 1] = 1

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed")
                    exit(1)
                }
                var stream: String = ""
                var streamBuffer: [CChar] = []
                var i_batch = batch.n_tokens - 1

                var n_cur = batch.n_tokens
                var n_decode = 0

                let t_main_start = ggml_time_us()

                while n_cur <= n_len {
                    if Task.isCancelled {
                        break
                    }
                    // prepare the next batch
                    batch.n_tokens = 0

                    // sample the next token for each parallel sequence / stream
                    if i_batch < 0 {
                        // the stream has already finished
                        break
                    }

                    let n_vocab = llama_n_vocab(self.model._model)
                    let logits = llama_get_logits_ith(context, i_batch)

                    var candidates: [llama_token_data] = .init(repeating: llama_token_data(), count: Int(n_vocab))

                    for token_id in 0 ..< n_vocab {
                        candidates.append(llama_token_data(id: token_id, logit: logits![Int(token_id)], p: 0.0))
                    }

                    var candidates_p: llama_token_data_array = .init(
                        data: &candidates,
                        size: candidates.count,
                        sorted: false
                    )

                    let top_k: Int32 = 40
                    let top_p: Float = 0.9
                    let temp: Float = 0.6

                    llama_sample_top_k(context, &candidates_p, top_k, 1)
                    llama_sample_top_p(context, &candidates_p, top_p, 1)
                    llama_sample_temp(context, &candidates_p, temp)

                    let new_token_id = llama_sample_token(context, &candidates_p)

                    // const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

                    // is it an end of stream? -> mark the stream as finished
                    if new_token_id == llama_token_eos(self.model._model) || n_cur == n_len {
                        i_batch = -1
                        continue
                    }

                    let nextStringPiece = self.token_to_piece(token: new_token_id, buffer: &streamBuffer) ?? ""
                    print(nextStringPiece, terminator: "")
                    stream += nextStringPiece
                    // yield to the continuation
                    continuation.yield(nextStringPiece)

                    // push this new token for next evaluation
                    batch.token[Int(batch.n_tokens)] = new_token_id
                    batch.pos[Int(batch.n_tokens)] = n_cur
                    batch.n_seq_id[Int(batch.n_tokens)] = 1
                    if let seq_id = batch.seq_id[Int(batch.n_tokens)] {
                        seq_id[0] = Int32(0)
                    }
                    batch.logits[Int(batch.n_tokens)] = 1

                    i_batch = batch.n_tokens

                    batch.n_tokens += 1

                    n_decode += 1

                    // all streams are finished
                    if batch.n_tokens == 0 {
                        break
                    }

                    n_cur += 1

                    // evaluate the current batch with the transformer model
                    if llama_decode(context, batch) != 0 {
                        print("llama_decode() failed")
                        exit(1)
                    }
                }

                let t_main_end = ggml_time_us()

                print("decoded \(n_decode) tokens in \(String(format: "%.2f", Double(t_main_end - t_main_start) / 1_000_000.0)) s, speed: \(String(format: "%.2f", Double(n_decode) / (Double(t_main_end - t_main_start) / 1_000_000.0))) t/s\n")

                llama_print_timings(context)
                continuation.finish()
            }
        }
    }
}


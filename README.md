# SwiftLlamaApp

This is a sample app built on the llama.cpp project.

https://github.com/ggerganov/llama.cpp

As llama.cpp is a Swift package, this app simply depends on the llama.cpp, and utilizes its API.

By default, tinyllama and ELYZA japanese Llama is set as a resource. Please re-link the model weight in the `SwiftLlamaApp/Resources/models` directory. 

You can downloade the model weight from the following URLs.

https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf

https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/tree/main

There is a blog entry for this sample.
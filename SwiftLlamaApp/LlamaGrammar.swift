//
//  LlamaGrammar.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/12/16.
//

import Foundation
import llama
import LlamaHelpers

final class LlamaGrammar {
    static var json: LlamaGrammar? {
        Self(#"""
    root   ::= object
    value  ::= object | array | string | number | ("true" | "false" | "null") ws

    object ::=
    "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
    )? "}" ws

    array  ::=
    "[" ws (
            value
    ("," ws value)*
    )? "]" ws

    string ::=
    "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
    )* "\"" ws

    number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

    # Optional space: by convention, applied in this grammar after literal chars when allowed
    ws ::= ([ \t\n] ws)?
    """#)
    }
    static var japanese_chat: LlamaGrammar? {
        Self(#"""
    root             ::= japanese-chat+
    japanese-chat    ::= ai-message | user-message | "\n"
    ai-message       ::= "Alan:" message
    user-message     ::= "User:" message
    message          ::= jp-char+ ([ \t\n] jp-char+)*
    jp-char          ::= hiragana | katakana | punctuation | cjk
    hiragana    ::= [ぁ-ゟ]
    katakana    ::= [ァ-ヿ]
    punctuation ::= [、-〾]
    cjk         ::= [一-鿿]
    """#)
    }

    var grammar: OpaquePointer

    init?(_ grammar: String) {
        print(grammar)
        self.grammar = grammar_parser.llama_grammar_init_from_content(grammar.cString(using: .utf8))
    }

    deinit {
        llama_grammar_free(self.grammar)
    }
}

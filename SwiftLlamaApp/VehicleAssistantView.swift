//
//  VehicleAssistantView.swift
//  SwiftLlamaApp
//
//  Created by miwa on 2023/12/19.
//

import SwiftUI

private struct AssistantResponse: Decodable {
    init(message: String, command: any AssistantCommand) {
        self.message = message
        self.command = command
    }
    
    var message: String
    var command: any AssistantCommand

    private enum CodingKeys: CodingKey {
        case message
        case command
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let message = try container.decode(String.self, forKey: .message)
        if let result = Self.tryDecode(container: container, command: SetAudioVolumeCommand.self, SetHvacTemperatureCommand.self, SetWindowLockCommand.self) {
            self.init(message: message, command: result)
        } else {
            throw DecodingError.dataCorruptedError(forKey: CodingKeys.command, in: container, debugDescription: "failed to decode command")
        }
    }

    // use error for reporting success (hack)
    private enum SuccessReport: Error {
        case success
    }
    private static func tryDecode<each Command: AssistantCommand>(
        container: KeyedDecodingContainer<CodingKeys>,
        command: repeat (each Command).Type
    ) -> (any AssistantCommand)? {
        func tryDecodeOne<C: AssistantCommand>(container: KeyedDecodingContainer<CodingKeys>, command _: C.Type, target: inout (any AssistantCommand)?) throws {
            do {
                target = try container.decode(C.self, forKey: .command)
            } catch {
                // do nothing
                return
            }
            throw SuccessReport.success
        }
        var result: (any AssistantCommand)? = nil
        do {
            repeat try tryDecodeOne(container: container, command: each command, target: &result)
        } catch SuccessReport.success {
            return result!
        } catch {
            fatalError("Unknown Error \(error)")
        }
        return nil
    }
}

private protocol AssitantCommandArgument: Decodable {
    static var typeDescription: String { get }
}

extension Int: AssitantCommandArgument {
    static var typeDescription: String { "number" }
}

extension Double: AssitantCommandArgument {
    static var typeDescription: String { "float" }
}

extension Bool: AssitantCommandArgument {
    static var typeDescription: String { "bool" }
}

private protocol AssistantCommand: Decodable {
    associatedtype Argument: AssitantCommandArgument
    static var name: String { get }
    var value: Argument { get }
    init(value: Argument)
    var description: String { get }
}

enum AssistantCommandCodingKeys: CodingKey {
    case valueType
    case value
    case commandName
}

enum AssistantCommandDecodeError: Error {
    case valueTypeUnmatched
    case commandNameUnmatched
}

extension AssistantCommand {
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AssistantCommandCodingKeys.self)
        let valueType = try container.decode(String.self, forKey: .valueType)
        guard valueType == Argument.typeDescription else {
            throw AssistantCommandDecodeError.valueTypeUnmatched
        }
        let commandName = try container.decode(String.self, forKey: .commandName)
        guard commandName == Self.name else {
            throw AssistantCommandDecodeError.commandNameUnmatched
        }
        self.init(value: try container.decode(Argument.self, forKey: .value))
    }
}

private struct SetAudioVolumeCommand: AssistantCommand {
    static let name = "AUDIO_VOLUME_SET_ABSOLUTE"
    var value: Int
    var description: String {
        return "\(Self.name)[\(value)]"
    }
}

private struct SetHvacTemperatureCommand: AssistantCommand {
    static let name = "HVAC_TEMPERATURE_SET_RELATIVE"
    var value: Double
    var description: String {
        return "\(Self.name)[\(value)]"
    }
}

private struct SetWindowLockCommand: AssistantCommand {
    static let name = "VEHICLE_LOCK_WINDOW"
    var value: Bool
    var description: String {
        return "\(Self.name)[\(value)]"
    }
}

struct VehicleAssistantView: View {
    @State private var request: String = ""
    @StateObject private var model = LlamaState()
    @State private var assistantResponse: AssistantResponse?
    @State private var inProgress = false
    var body: some View {
        TextField("リクエスト", text: $request)
        Button("よろしく！") {
            Task {
                self.inProgress = true
                self.assistantResponse = await getResponse(request: request)
                self.inProgress = false
            }
        }
        if let assistantResponse {
            Text("Request: " + request)
            Text("Message: " + assistantResponse.message)
            Text("Command: " + assistantResponse.command.description)
        } else if inProgress {
            ProgressView()
        }
    }

    private func getResponse(request: String) async -> AssistantResponse? {
        guard let jsonGrammar = LlamaGrammar.json else {
            return nil
        }
        guard let result = try? await self.model.generateWithGrammar(prompt: """
        The data is request by user and response of in-vehicle infortainment AI assitant. AI assistant can use following commands; "AUDIO_VOLUME_SET_ABSOLUTE" (arg: 0<number<100) / "HVAC_TEMPERATURE_SET_RELATIVE" (arg: float, positive is warmer) / "VEHICLE_LOCK_WINDOW" (arg: bool, true is locked).
        There must be message and command. command includes commandName, value, and valueType.

        req: "Please lock the windows"
        res: {"message": "sure, windows are now locked.", "command": {"commandName": "VEHICLE_LOCK_WINDOW", "value": true, "valueType": "bool"}}

        req: "It's too hot!"
        res: {"message": "I'm sorry, I'll lower the temperature soon.", "command": {"commandName": "HVAC_TEMPERATURE_SET_RELATIVE", "value": -2.0, "valueType": "float"}}

        req: "\(request)"
        res:
        """, grammar: jsonGrammar) else {
            return nil
        }
        print(result)
        return try? JSONDecoder().decode(AssistantResponse.self, from: result.data(using: .utf8)!)
    }
}

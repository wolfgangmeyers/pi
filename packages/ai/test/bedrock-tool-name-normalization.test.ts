import { Type } from "@sinclair/typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";

type SendResult = { kind: "reject"; error: unknown } | { kind: "resolve"; response: unknown };

const bedrockMock = vi.hoisted(() => ({
	lastInput: undefined as unknown,
	send: undefined as SendResult | undefined,
}));

vi.mock("@aws-sdk/client-bedrock-runtime", () => {
	class BedrockRuntimeServiceException extends Error {}

	class BedrockRuntimeClient {
		middlewareStack = { add: () => {} };

		send(command: { input: unknown }): Promise<unknown> {
			bedrockMock.lastInput = command.input;
			const outcome = bedrockMock.send;
			if (!outcome) return Promise.reject(new Error("test did not configure a send outcome"));
			return outcome.kind === "reject" ? Promise.reject(outcome.error) : Promise.resolve(outcome.response);
		}
	}

	class ConverseStreamCommand {
		readonly input: unknown;
		constructor(input: unknown) {
			this.input = input;
		}
	}

	return {
		BedrockRuntimeClient,
		BedrockRuntimeServiceException,
		ConverseStreamCommand,
		StopReason: {
			END_TURN: "end_turn",
			STOP_SEQUENCE: "stop_sequence",
			MAX_TOKENS: "max_tokens",
			MODEL_CONTEXT_WINDOW_EXCEEDED: "model_context_window_exceeded",
			TOOL_USE: "tool_use",
		},
		CachePointType: { DEFAULT: "default" },
		CacheTTL: { ONE_HOUR: "ONE_HOUR" },
		ConversationRole: { ASSISTANT: "assistant", USER: "user" },
		ImageFormat: { JPEG: "jpeg", PNG: "png", GIF: "gif", WEBP: "webp" },
		ToolResultStatus: { ERROR: "error", SUCCESS: "success" },
	};
});

import { stream as streamBedrock } from "../src/api/bedrock-converse-stream.ts";
import { getModel } from "../src/compat.ts";
import type { AssistantMessage, Context, Tool } from "../src/types.ts";

const model = getModel("amazon-bedrock", "zai.glm-5");
const parameters = Type.Object({ value: Type.String() });

function tool(name: string): Tool {
	return { name, description: `Run ${name}`, parameters };
}

async function run(context: Context): Promise<AssistantMessage> {
	return streamBedrock(model, context, { cacheRetention: "none" }).result();
}

beforeEach(() => {
	bedrockMock.lastInput = undefined;
	bedrockMock.send = {
		kind: "resolve",
		response: {
			$metadata: { httpStatusCode: 200, requestId: "request-id" },
			stream: (async function* () {
				yield { messageStart: { role: "assistant" } };
				yield { messageStop: { stopReason: "end_turn" } };
			})(),
		},
	};
});

describe("bedrock tool name normalization", () => {
	it("normalizes invalid outbound tool names for Bedrock", async () => {
		await run({
			messages: [{ role: "user", content: "hello", timestamp: 0 }],
			tools: [tool("multi_tool_use.parallel")],
		});

		expect(bedrockMock.lastInput).toMatchObject({
			toolConfig: { tools: [{ toolSpec: { name: "multi_tool_use_parallel" } }] },
		});
	});

	it("restores provider tool calls to original tool names", async () => {
		bedrockMock.send = {
			kind: "resolve",
			response: {
				$metadata: { httpStatusCode: 200, requestId: "request-id" },
				stream: (async function* () {
					yield { messageStart: { role: "assistant" } };
					yield {
						contentBlockStart: {
							contentBlockIndex: 0,
							start: { toolUse: { toolUseId: "tool-1", name: "multi_tool_use_parallel" } },
						},
					};
					yield { contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input: '{"value":"ok"}' } } } };
					yield { contentBlockStop: { contentBlockIndex: 0 } };
					yield { messageStop: { stopReason: "tool_use" } };
				})(),
			},
		};

		const message = await run({
			messages: [{ role: "user", content: "use the tool", timestamp: 0 }],
			tools: [tool("multi_tool_use.parallel")],
		});

		expect(message.stopReason).toBe("toolUse");
		expect(message.content).toContainEqual({
			type: "toolCall",
			id: "tool-1",
			name: "multi_tool_use.parallel",
			arguments: { value: "ok" },
		});
	});

	it("normalizes replayed assistant tool calls and explicit tool choices", async () => {
		await run({
			messages: [
				{ role: "user", content: "hello", timestamp: 0 },
				{
					role: "assistant",
					content: [
						{ type: "toolCall", id: "tool-1", name: "multi_tool_use.parallel", arguments: { value: "ok" } },
					],
					timestamp: 1,
				},
			],
			tools: [tool("multi_tool_use.parallel")],
		});

		const replayedAssistant = (bedrockMock.lastInput as { messages: Array<{ role?: string; content?: unknown[] }> }).messages.find(
			(message) => message.role === "assistant",
		);
		expect(replayedAssistant?.content).toContainEqual({
			toolUse: { name: "multi_tool_use_parallel", toolUseId: "tool-1", input: { value: "ok" } },
		});

		await streamBedrock(
			model,
			{ messages: [{ role: "user", content: "hello", timestamp: 0 }], tools: [tool("multi_tool_use.parallel")] },
			{ cacheRetention: "none", toolChoice: { type: "tool", name: "multi_tool_use.parallel" } },
		).result();

		expect(bedrockMock.lastInput).toMatchObject({
			toolConfig: { toolChoice: { tool: { name: "multi_tool_use_parallel" } } },
		});
	});

	it("fails locally when different tool names normalize to the same Bedrock name", async () => {
		const message = await run({
			messages: [{ role: "user", content: "hello", timestamp: 0 }],
			tools: [tool("a.b"), tool("a_b")],
		});

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toContain('Bedrock tool names "a.b" and "a_b" both normalize to "a_b"');
	});

	it("fails locally when explicit tool choice references an unknown tool name", async () => {
		const message = await streamBedrock(
			model,
			{ messages: [{ role: "user", content: "hello", timestamp: 0 }], tools: [tool("safe_tool")] },
			{ cacheRetention: "none", toolChoice: { type: "tool", name: "multi_tool_use.parallel" } },
		).result();

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toContain('Bedrock tool choice references unknown tool name "multi_tool_use.parallel"');
	});
});

import { Type } from "typebox";
import { afterEach, describe, expect, test, vi } from "vitest";
import { TOOL_SEARCH_TRAILING_NOTE } from "../src/api/openai-responses-shared.ts";
import { getModel, streamSimple } from "../src/compat.ts";
import type { Api, Context, Message, Model, SystemMessage, Tool } from "../src/types.ts";
import { resolveTranscriptTools } from "../src/utils/transcript.ts";

class PayloadCaptured extends Error {}

afterEach(() => {
	vi.restoreAllMocks();
});

/** A complete Responses SSE body, so a test can run the real adapter and keep its assistant message. */
function completedResponsesResponse(): Response {
	const event = {
		type: "response.completed",
		sequence_number: 0,
		response: {
			id: "resp_trailing_note",
			status: "completed",
			output: [],
			usage: {
				input_tokens: 1,
				output_tokens: 1,
				total_tokens: 2,
				input_tokens_details: { cached_tokens: 0 },
			},
		},
	};
	return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

/** The Responses model whose compat makes the converter emit the trailing tool note. */
function trailingNoteModel(): Model<"openai-responses"> {
	return {
		...modelBase,
		id: "deepseek-flash",
		name: "DeepSeek V4.1 Flash",
		api: "openai-responses",
		provider: "deepseek",
		compat: {
			supportsDeveloperRole: false,
			supportsMidConvoSystemMessages: true,
			supportsToolSearch: true,
			requiresToolSearchTrailingNote: true,
		},
	};
}

function tool(name: string): Tool {
	return { name, description: `${name} tool`, parameters: Type.Object({}) };
}

async function capturePayload<T>(model: Model<Api>, context: Context): Promise<T> {
	let captured: T | undefined;
	const stream = streamSimple(model, context, {
		apiKey: "test-key",
		onPayload: (payload) => {
			captured = payload as T;
			throw new PayloadCaptured();
		},
	});
	await stream.result();
	if (!captured) throw new Error("Expected payload capture");
	return captured;
}

const modelBase = {
	baseUrl: "http://127.0.0.1:9",
	reasoning: true,
	input: ["text"] as ("text" | "image")[],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 1000,
};

const baseTool = tool("base_tool");
const lateTool = tool("late_tool");
const dropTool = tool("drop_tool");

/**
 * Capture one request after each step: initial tools, removal of drop_tool, then two loads.
 * Returns the payloads after the removal and after each load, plus the one before the removal.
 */
async function captureToolHistory<T>(model: Model<Api>): Promise<T[]> {
	const steps: Message[][] = [
		[{ role: "user", content: "first", timestamp: 1 }],
		[
			{ role: "system", content: "", toolsRemoved: [{ name: "drop_tool" }], timestamp: 2 },
			{ role: "user", content: "after removal", timestamp: 3 },
		],
		[
			{ role: "system", content: "", toolsAdded: [lateTool], timestamp: 4 },
			{ role: "user", content: "after first load", timestamp: 5 },
		],
		[
			{ role: "system", content: "", toolsAdded: [tool("later_tool")], timestamp: 6 },
			{ role: "user", content: "after second load", timestamp: 7 },
		],
	];
	const messages: Message[] = [
		{ role: "system", content: "base prompt", toolsAdded: [baseTool, dropTool], timestamp: 0 },
	];
	const payloads: T[] = [];
	for (const step of steps) {
		messages.push(...step);
		payloads.push(await capturePayload<T>(model, { messages: [...messages] }));
	}
	return payloads;
}

/** Every request's message list starts with the previous request's message list. */
function expectGrowingPrefix(inputs: readonly unknown[][]): void {
	for (let index = 1; index < inputs.length; index++) {
		const previous = inputs[index - 1] ?? [];
		expect(inputs[index]?.slice(0, previous.length)).toEqual(previous);
	}
}

type ResponsesPayload = {
	tools?: Array<{ name: string; description?: string }>;
	input: Array<{ type?: string; role?: string; content?: string; tools?: Array<{ name: string }> }>;
};

const additionalToolsModel: Model<"openai-responses"> = {
	...modelBase,
	id: "gpt-5.4",
	name: "GPT-5.4",
	api: "openai-responses",
	provider: "openai",
	compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
};
const context: Context = {
	messages: [
		{
			role: "system",
			content: "base prompt",
			sections: { rules: "<rules>\nold rules\n</rules>", docs: "<docs>\nread docs\n</docs>" },
			toolsAdded: [baseTool],
			timestamp: 0,
		},
		{ role: "user", content: "before", timestamp: 1 },
		{
			role: "system",
			content: "updated guidance",
			sections: { rules: "<rules>\nnew rules\n</rules>", docs: null },
			toolsRemoved: [{ name: "base_tool" }],
			toolsAdded: [lateTool],
			timestamp: 2,
		},
	],
};
const additionContext: Context = {
	messages: [
		{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
		{ role: "user", content: "before", timestamp: 1 },
		{ role: "system", content: "updated guidance", toolsAdded: [lateTool], timestamp: 2 },
	],
};

const anthropicNativeModel: Model<"anthropic-messages"> = {
	...modelBase,
	id: "claude-opus-5",
	name: "Claude Opus 5",
	api: "anthropic-messages",
	provider: "anthropic",
	compat: { supportsMidConvoSystemMessages: true, supportsMidConvoToolChanges: true },
};

interface AnthropicToolDefinition {
	name: string;
	description?: string;
	defer_loading?: boolean;
	cache_control?: unknown;
}

interface AnthropicPayload {
	betas?: string[];
	system?: Array<{ text: string }>;
	tools?: AnthropicToolDefinition[];
	messages: Array<{
		role: string;
		content: Array<{
			type: string;
			text?: string;
			cache_control?: unknown;
			tool?: { type: string; name?: string; definition?: AnthropicToolDefinition };
		}>;
	}>;
}

describe("transcript system messages", () => {
	test("sends Anthropic updates and tool changes in native system messages", async () => {
		const payload = await capturePayload<AnthropicPayload>(anthropicNativeModel, context);

		expect(payload.betas).toContain("inline-tools-2026-09-15");
		expect(payload.betas).not.toContain("mid-conversation-tool-changes-2026-07-01");
		expect(payload.system?.map((block) => block.text)).toEqual([
			"base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>",
		]);
		// The top-level list holds only the initial tools (with the cache breakpoint) and the
		// deferred placeholder; later tools never appear there.
		expect(payload.tools).toMatchObject([
			{ name: "base_tool", cache_control: { type: "ephemeral" } },
			{ name: "__pi_deferred_placeholder__", defer_loading: true },
		]);
		expect(payload.tools).toHaveLength(2);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
		expect(payload.tools?.[1]?.cache_control).toBeUndefined();
		const update = payload.messages.at(-1);
		expect(update).toMatchObject({
			role: "system",
			content: [
				{ type: "text" },
				{ type: "tool_removal", tool: { type: "tool_reference", name: "base_tool" } },
				{
					type: "tool_addition",
					tool: { type: "tool_definition", definition: { name: "late_tool", description: "late_tool tool" } },
					cache_control: { type: "ephemeral" },
				},
			],
		});
		// Cache control goes on the block, never on the inline definition; nothing is deferred.
		const definition = update?.content[2]?.tool?.definition;
		expect(definition?.cache_control).toBeUndefined();
		expect(definition?.defer_loading).toBeUndefined();
		expect(update?.content[0]?.text).toContain("updated guidance");
		expect(update?.content[0]?.text).toContain("<rules>\nnew rules\n</rules>");
		expect(update?.content[0]?.text).toContain('Removed system prompt section "docs"');

		// The placeholder is declared before any change so its scaffolding is cached from request one.
		const initial = await capturePayload<AnthropicPayload>(anthropicNativeModel, {
			messages: context.messages.slice(0, 2),
		});
		expect(initial.tools?.map((tool) => tool.name)).toEqual(["base_tool", "__pi_deferred_placeholder__"]);
	});

	test("redefines an Anthropic tool by value under the same name", async () => {
		const redefinedTool = { ...baseTool, description: "changed" };
		const payload = await capturePayload<AnthropicPayload>(anthropicNativeModel, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{
					role: "system",
					content: "",
					toolsRemoved: [{ name: "base_tool" }],
					toolsAdded: [redefinedTool],
					timestamp: 2,
				},
			],
		});

		expect(payload.betas).toContain("inline-tools-2026-09-15");
		// The top-level declaration keeps the original definition, so the cached prefix survives.
		expect(payload.tools).toMatchObject([
			{ name: "base_tool", description: "base_tool tool" },
			{ name: "__pi_deferred_placeholder__" },
		]);
		// The new definition replaces the old one in place; no removal block is sent for it.
		expect(payload.messages.at(-1)).toMatchObject({
			role: "system",
			content: [
				{
					type: "tool_addition",
					tool: { type: "tool_definition", definition: { name: "base_tool", description: "changed" } },
				},
			],
		});
		expect(payload.messages.at(-1)?.content).toHaveLength(1);
	});

	test("sends the current Anthropic tool list without an initial tool", async () => {
		// Anthropic rejects a tool list where every tool (here: the placeholder) is deferred.
		const payload = await capturePayload<AnthropicPayload>(anthropicNativeModel, {
			messages: [
				{ role: "system", content: "base prompt", timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{ role: "system", content: "updated guidance", toolsAdded: [lateTool], timestamp: 2 },
			],
		});
		expect(payload.betas ?? []).not.toContain("inline-tools-2026-09-15");
		expect(payload.tools).toMatchObject([{ name: "late_tool", cache_control: { type: "ephemeral" } }]);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
		expect(payload.messages.at(-1)?.content.map((block) => block.type)).toEqual(["text"]);
	});

	test("folds Anthropic updates into the system prompt without native support", async () => {
		const model: Model<"anthropic-messages"> = {
			...modelBase,
			id: "claude-sonnet-4-5",
			name: "Claude Sonnet 4.5",
			api: "anthropic-messages",
			provider: "anthropic",
		};
		const payload = await capturePayload<{
			betas?: string[];
			system?: Array<{ text: string }>;
			tools?: Array<{ name: string }>;
			messages: Array<{ role: string }>;
		}>(model, context);

		expect(payload.betas ?? []).not.toContain("inline-tools-2026-09-15");
		expect(payload.system?.map((block) => block.text)).toEqual([
			"base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>",
		]);
		expect(payload.tools?.map((value) => value.name)).toEqual(["late_tool"]);
		expect(payload.messages.map((message) => message.role)).toEqual(["user"]);
	});

	test("requires both Anthropic capabilities for native tool changes", async () => {
		const model: Model<"anthropic-messages"> = {
			...modelBase,
			id: "claude-opus-5",
			name: "Claude Opus 5",
			api: "anthropic-messages",
			provider: "anthropic",
			compat: { supportsMidConvoToolChanges: true },
		};
		const payload = await capturePayload<{
			betas?: string[];
			tools?: Array<{ name: string }>;
			messages: Array<{ role: string }>;
		}>(model, context);

		expect(payload.betas ?? []).not.toContain("inline-tools-2026-09-15");
		expect(payload.tools?.map((value) => value.name)).toEqual(["late_tool"]);
		expect(payload.messages.map((message) => message.role)).toEqual(["user"]);
	});

	test("declares every OpenAI tool in additional_tools at its developer message", async () => {
		const payload = await capturePayload<ResponsesPayload>(additionalToolsModel, additionContext);

		// A later additional_tools item cannot replace a top-level tool, so no tool is declared there.
		expect(payload.tools).toBeUndefined();
		expect(
			payload.input.filter((item) => item.type === "additional_tools").map((item) => item.tools?.[0]?.name),
		).toEqual(["base_tool", "late_tool"]);
		// The incremental notice keeps models from reading a later item as the complete tool set.
		expect(payload.input.map((item) => item.type ?? item.content)).toEqual([
			"additional_tools",
			"base prompt",
			expect.any(Array),
			expect.stringContaining("incremental tools update"),
			"additional_tools",
			"updated guidance",
		]);
	});

	test("maps system-message additions into synthetic tool search", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsToolSearch: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; tools?: Array<{ name: string }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expect(payload.input.map((item) => item.type)).toContain("tool_search_call");
		expect(
			payload.input.find((item) => item.type === "tool_search_output")?.tools?.map((value) => value.name),
		).toEqual(["late_tool"]);
	});

	test("keeps the trailing note on every request that re-emits the tool-search pair", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "deepseek-flash",
			name: "DeepSeek V4.1 Flash",
			api: "openai-responses",
			provider: "deepseek",
			compat: {
				supportsDeveloperRole: false,
				supportsMidConvoSystemMessages: true,
				supportsToolSearch: true,
				requiresToolSearchTrailingNote: true,
			},
		};
		type Payload = {
			tools?: Array<{ name: string }>;
			input: Array<{
				type?: string;
				role?: string;
				content?: string;
				call_id?: string;
				tools?: Array<{ name: string }>;
			}>;
		};
		const current = await capturePayload<Payload>(model, additionContext);
		const followup = await capturePayload<Payload>(model, {
			messages: [...additionContext.messages, { role: "user", content: "after", timestamp: 3 }],
		});
		const systemAfter = await capturePayload<Payload>(model, {
			messages: [...additionContext.messages, { role: "system", content: "later guidance", timestamp: 3 }],
		});
		for (const payload of [current, followup, systemAfter]) {
			expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
			expect(
				payload.input.find((item) => item.type === "tool_search_output")?.tools?.map((tool) => tool.name),
			).toEqual(["late_tool"]);
		}
		const trailingNote = { role: "developer", content: TOOL_SEARCH_TRAILING_NOTE };
		// The pair rides every later request, so the note must too: a provider that requires a non-assistant
		// item after the pair rejects a request whose tail is only assistant/tool items, and DeepSeek's
		// Responses endpoint is exactly that provider (probed live 2026-10-08). Nothing in the wording
		// claims a fresh change, so re-sending it on a request that merely re-emits the pair reports nothing
		// false.
		for (const payload of [current, followup, systemAfter]) {
			expect(payload.input.at(-1)).toEqual(trailingNote);
			expect(
				payload.input.filter((item) => item.role === "developer" && item.content === TOOL_SEARCH_TRAILING_NOTE),
			).toHaveLength(1);
		}
		// A system checkpoint that follows the addition is still a request that re-emits the pair.
		expect(followup.input.at(-2)).toEqual({ role: "user", content: [{ type: "input_text", text: "after" }] });

		const pair = (payload: Payload) => payload.input.findIndex((item) => item.type === "tool_search_call");
		const output = (payload: Payload) => payload.input.findIndex((item) => item.type === "tool_search_output");
		const notedCallId = current.input[pair(current)]?.call_id;
		expect(pair(current)).toBeGreaterThanOrEqual(0);
		expect(output(current)).toBeGreaterThanOrEqual(0);
		expect(typeof notedCallId).toBe("string");
		expect(notedCallId).not.toBe("");
		expect(current.input[output(current)]?.call_id).toBe(notedCallId);
		expect(pair(followup)).toBe(pair(current));
		expect(followup.input[pair(followup)]?.call_id).toBe(notedCallId);
	});

	test("keeps the trailing note truthful when a later update removes the added tool", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "deepseek-flash",
			name: "DeepSeek V4.1 Flash",
			api: "openai-responses",
			provider: "deepseek",
			compat: {
				supportsDeveloperRole: false,
				supportsMidConvoSystemMessages: true,
				supportsToolSearch: true,
				requiresToolSearchTrailingNote: true,
			},
		};
		type Payload = { input: Array<{ type?: string; role?: string; content?: string }> };
		// The pair is already in the transcript, so it keeps riding; the removal arrives as its own update.
		const removed = await capturePayload<Payload>(model, {
			messages: [
				...additionContext.messages,
				{ role: "system", content: "", toolsRemoved: [{ name: "late_tool" }], timestamp: 3 },
			],
		});
		expect(removed.input.some((item) => item.type === "tool_search_call")).toBe(true);
		expect(removed.input.at(-1)).toEqual({ role: "developer", content: TOOL_SEARCH_TRAILING_NOTE });
		// The note rides requests the removal has already superseded, so it must not promise callability the
		// transcript has since taken away. The tool-set patch above it is what reports the removal.
		expect(TOOL_SEARCH_TRAILING_NOTE).not.toMatch(/(is|are) callable now/);
	});

	test("records the trailing note on every turn whose request carried it", async () => {
		const model = trailingNoteModel();
		vi.spyOn(globalThis, "fetch").mockImplementation(async () => completedResponsesResponse());

		// The note is a rendering of one request, so each turn that received it is where it gets recorded.
		const addition = await streamSimple(model, additionContext, { apiKey: "test-key" }).result();
		// A later request re-emits the tool-search pair and therefore sends the note again.
		const later = await streamSimple(
			model,
			{ messages: [...additionContext.messages, { role: "user", content: "after", timestamp: 3 }] },
			{ apiKey: "test-key" },
		).result();

		const noteDiagnostic = [
			{
				type: "tool_search_trailing_note",
				timestamp: expect.any(Number),
				details: { note: TOOL_SEARCH_TRAILING_NOTE },
			},
		];
		expect(addition.diagnostics).toEqual(noteDiagnostic);
		expect(later.diagnostics).toEqual(noteDiagnostic);
	});

	test("records the note when the request that transmitted it failed", async () => {
		const model = trailingNoteModel();
		vi.spyOn(globalThis, "fetch").mockImplementation(
			async () => new Response("provider exploded", { status: 500, headers: { "content-type": "text/plain" } }),
		);

		// The note rode a real request, so the turn records it even though the turn itself failed: the
		// transcript's own stopReason is what tells the reader the turn did not complete.
		const failed = await streamSimple(model, additionContext, { apiKey: "test-key", maxRetries: 0 }).result();

		expect(failed.stopReason).toBe("error");
		expect(failed.diagnostics?.map((diagnostic) => diagnostic.type)).toEqual(["tool_search_trailing_note"]);
	});

	test("records a note that a payload handler pushed off the end", async () => {
		const model = trailingNoteModel();
		let sentInput: Array<{ role?: string; content?: unknown }> = [];
		vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
			sentInput = (JSON.parse(String(init?.body)) as { input: typeof sentInput }).input;
			return completedResponsesResponse();
		});

		const rewritten = await streamSimple(model, additionContext, {
			apiKey: "test-key",
			onPayload: (payload) => {
				const input = (payload as { input: unknown[] }).input;
				return {
					...(payload as object),
					input: [...input, { role: "developer", content: "appended after the note" }],
				};
			},
		}).result();

		// The note was transmitted, just no longer last, so the record must follow the wire and not the tail.
		expect(sentInput.some((item) => item.role === "developer" && item.content === TOOL_SEARCH_TRAILING_NOTE)).toBe(
			true,
		);
		expect(rewritten.diagnostics?.map((diagnostic) => diagnostic.type)).toEqual(["tool_search_trailing_note"]);
	});

	test("records nothing when the turn was already cancelled", async () => {
		const model = trailingNoteModel();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => completedResponsesResponse());
		const controller = new AbortController();
		controller.abort();

		// No request left the machine, so there is no request for the record to describe.
		const cancelled = await streamSimple(model, additionContext, {
			apiKey: "test-key",
			signal: controller.signal,
		}).result();

		expect(fetchSpy).not.toHaveBeenCalled();
		expect(cancelled.stopReason).toBe("aborted");
		expect(cancelled.diagnostics).toBeUndefined();
	});

	test("records a note a payload handler rewrote into content parts", async () => {
		const model = trailingNoteModel();
		let sentInput: unknown[] = [];
		vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
			sentInput = (JSON.parse(String(init?.body)) as { input: unknown[] }).input;
			return completedResponsesResponse();
		});

		const rewritten = await streamSimple(model, additionContext, {
			apiKey: "test-key",
			onPayload: (payload) => {
				const input = (payload as { input: unknown[] }).input;
				return {
					...(payload as object),
					input: input.map((item) => {
						const candidate = item as { role?: string; content?: unknown };
						return candidate.role === "developer" && candidate.content === TOOL_SEARCH_TRAILING_NOTE
							? { ...candidate, content: [{ type: "input_text", text: TOOL_SEARCH_TRAILING_NOTE }] }
							: item;
					}),
				};
			},
		}).result();

		// The instruction still reached the wire, so the record must follow the text, not its encoding.
		expect(JSON.stringify(sentInput)).toContain(TOOL_SEARCH_TRAILING_NOTE);
		expect(rewritten.diagnostics?.map((diagnostic) => diagnostic.type)).toEqual(["tool_search_trailing_note"]);
	});

	test("folds OpenAI updates into the leading developer message without native support", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-4.1",
			name: "GPT-4.1",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsAdditionalTools: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; role?: string; content?: string }>;
		}>(model, context);

		expect(payload.tools?.map((value) => value.name)).toEqual(["late_tool"]);
		expect(payload.input.map((item) => item.type ?? item.role)).toEqual(["developer", "user"]);
		expect(payload.input[0]?.content).toBe("base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>");
	});

	test("announces OpenAI removals as text", async () => {
		const payload = await capturePayload<ResponsesPayload>(additionalToolsModel, context);

		expect(payload.tools).toBeUndefined();
		expect(
			payload.input.filter((item) => item.type === "additional_tools").map((item) => item.tools?.[0]?.name),
		).toEqual(["base_tool", "late_tool"]);
		const update = payload.input.at(-1)?.content;
		expect(update).toContain("updated guidance");
		expect(update).toContain("The following tools are no longer available. Do not call them:\n- base_tool");
	});

	test("redefines an OpenAI tool with additional_tools instead of announcing a removal", async () => {
		const payload = await capturePayload<ResponsesPayload>(additionalToolsModel, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{
					role: "system",
					content: "",
					toolsRemoved: [{ name: "base_tool" }],
					toolsAdded: [{ ...baseTool, description: "changed" }],
					timestamp: 2,
				},
			],
		});

		expect(payload.tools).toBeUndefined();
		expect(payload.input).toMatchObject([
			{ type: "additional_tools", tools: [{ name: "base_tool", description: "base_tool tool" }] },
			{ role: "developer", content: "base prompt" },
			{ role: "user" },
			{ role: "developer", content: expect.stringContaining("latest definition replaces the earlier one") },
			{ type: "additional_tools", tools: [{ name: "base_tool", description: "changed" }] },
		]);
		expect(payload.input).toHaveLength(5);
	});

	// #10940: deferred loads after a removal must not rewrite the cached prefix.
	test("keeps the OpenAI request prefix across a removal and later loads", async () => {
		const payloads = await captureToolHistory<ResponsesPayload>(additionalToolsModel);

		for (const payload of payloads) expect(payload.tools).toBeUndefined();
		expectGrowingPrefix(payloads.map((payload) => payload.input));
		const last = payloads.at(-1);
		expect(
			last?.input.filter((item) => item.type === "additional_tools").map((item) => item.tools?.map((t) => t.name)),
		).toEqual([["base_tool", "drop_tool"], ["late_tool"], ["later_tool"]]);
		expect(
			last?.input.some((item) => item.content?.includes("no longer available. Do not call them:\n- drop_tool")),
		).toBe(true);
		// Each later load restates the tools that are still unavailable.
		expect(
			last?.input.filter((item) => item.content?.includes("incremental tools update")).map((item) => item.content),
		).toEqual([
			expect.stringContaining("\nStill unavailable: drop_tool."),
			expect.stringContaining("\nStill unavailable: drop_tool."),
		]);
	});

	test("stops listing a tool as unavailable once it is declared again", async () => {
		const payload = await capturePayload<ResponsesPayload>(additionalToolsModel, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool, dropTool], timestamp: 0 },
				{ role: "system", content: "", toolsRemoved: [{ name: "drop_tool" }], timestamp: 1 },
				{ role: "system", content: "", toolsAdded: [dropTool], timestamp: 2 },
				{ role: "user", content: "go", timestamp: 3 },
			],
		});

		const notice = payload.input.find((item) => item.content?.includes("incremental tools update"))?.content;
		expect(notice).toBeDefined();
		expect(notice).not.toContain("Still unavailable");
		expect(payload.input.at(-2)).toMatchObject({ type: "additional_tools", tools: [{ name: "drop_tool" }] });
	});

	test("does not list a tool as still unavailable before its removal is announced", async () => {
		const payload = await capturePayload<ResponsesPayload>(additionalToolsModel, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool, dropTool], timestamp: 0 },
				{
					role: "system",
					content: "",
					toolsAdded: [lateTool],
					toolsRemoved: [{ name: "drop_tool" }],
					timestamp: 1,
				},
				{ role: "user", content: "go", timestamp: 2 },
			],
		});

		// Like Codex: notice, then the tools, then the removal notice.
		const [notice] = payload.input.slice(-4);
		expect(notice?.content).toContain("incremental tools update");
		expect(notice?.content).not.toContain("Still unavailable");
		expect(payload.input.slice(-3)).toMatchObject([
			{ type: "additional_tools", tools: [{ name: "late_tool" }] },
			{ content: expect.stringContaining("no longer available. Do not call them:\n- drop_tool") },
			{ role: "user" },
		]);
	});

	// #10940: synthetic tool search cannot remove tools, so a removal rewrites the prefix once.
	test("keeps the OpenAI tool search prefix across loads after a removal", async () => {
		const model: Model<"openai-responses"> = {
			...additionalToolsModel,
			compat: { supportsMidConvoSystemMessages: true, supportsToolSearch: true },
		};
		const [beforeRemoval, ...afterRemoval] = await captureToolHistory<ResponsesPayload>(model);

		expect(beforeRemoval?.tools?.map((value) => value.name)).toEqual(["base_tool", "drop_tool"]);
		for (const payload of afterRemoval) expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expectGrowingPrefix(afterRemoval.map((payload) => payload.input));
		expect(
			afterRemoval
				.at(-1)
				?.input.filter((item) => item.type === "tool_search_output")
				.map((item) => item.tools?.[0]?.name),
		).toEqual(["late_tool", "later_tool"]);
	});

	// #10940: removals and redeclarations move the top-level boundary once; plain additions stay anchored.
	// The private change: a removal alone never moves the flat field, because the removed tool keeps
	// its declaration and slot and a call to it is refused rather than executed. Only a name declared
	// again moves it, including a re-add, which Kimi K3 otherwise ignores.
	test("keeps the flat tool field anchored when the only change is a removal", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsToolSearch: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; role?: string; tools?: Array<{ name: string }> }>;
		}>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{ role: "system", content: "", toolsRemoved: [{ name: "base_tool" }], timestamp: 2 },
				{ role: "user", content: "after", timestamp: 3 },
			],
		});

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expect(
			payload.input.filter((item) => item.type === "tool_search_output").map((item) => item.tools?.[0]?.name),
		).toEqual([]);
	});

	test("resolves the top-level tool boundary per transport", () => {
		const messages = [
			{ role: "system", content: "base prompt", toolsAdded: [baseTool, dropTool], timestamp: 0 },
			{ role: "system", content: "", toolsRemoved: [{ name: "drop_tool" }], timestamp: 1 },
			{ role: "system", content: "", toolsAdded: [dropTool], timestamp: 2 },
			{ role: "system", content: "", toolsAdded: [{ ...lateTool, description: "x" }], timestamp: 3 },
			{ role: "system", content: "", toolsAdded: [{ ...lateTool, description: "y" }], timestamp: 4 },
		] satisfies SystemMessage[];

		expect(resolveTranscriptTools(messages.slice(0, 2), "additions")).toEqual({
			requestTools: [baseTool],
			foldedSystemMessages: 2,
		});
		// Kimi K3 ignores a removed tool that a later tool message declares again, so the re-added
		// drop_tool moves the boundary; the new late_tool after it stays anchored.
		expect(resolveTranscriptTools(messages.slice(0, 4), "additions")).toEqual({
			requestTools: [baseTool, dropTool],
			foldedSystemMessages: 3,
		});
		// The redefinition of late_tool moves the boundary to its message.
		expect(resolveTranscriptTools(messages, "additions")).toEqual({
			requestTools: [baseTool, dropTool, { ...lateTool, description: "y" }],
			foldedSystemMessages: 5,
		});
		expect(resolveTranscriptTools(messages, "changes")).toEqual({ requestTools: [], foldedSystemMessages: 0 });
		expect(resolveTranscriptTools(messages, "none")).toEqual({
			requestTools: [baseTool, dropTool, { ...lateTool, description: "y" }],
			foldedSystemMessages: 5,
		});
	});

	test("keeps the declared list byte identical when a tool leaves and returns unchanged", async () => {
		const secondTool = tool("second_tool");
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
		};
		const before = await capturePayload<{ tools?: Array<{ name: string; description: string }> }>(model, {
			messages: [{ role: "system", content: "base prompt", toolsAdded: [baseTool, secondTool], timestamp: 0 }],
		});
		const after = await capturePayload<{ tools?: Array<{ name: string; description: string }> }>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool, secondTool], timestamp: 0 },
				{ role: "system", content: "", toolsRemoved: [{ name: "base_tool" }], timestamp: 1 },
				{ role: "system", content: "", toolsAdded: [baseTool], timestamp: 2 },
			],
		});

		expect(after.tools?.map((value) => value.name)).toEqual(["base_tool", "second_tool"]);
		expect(after.tools).toEqual(before.tools);
	});

	test("keeps first-declaration order when a tool is redefined", async () => {
		const secondTool = tool("second_tool");
		const redefinedBaseTool = { ...baseTool, description: "base_tool v2" };
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
		};
		const payload = await capturePayload<{ tools?: Array<{ name: string }> }>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool, secondTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{
					role: "system",
					content: "updated guidance",
					toolsRemoved: [{ name: "base_tool" }],
					toolsAdded: [redefinedBaseTool],
					timestamp: 2,
				},
			],
		});

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool", "second_tool"]);
	});

	test("uses the latest definition after a same-name tool change", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "deepseek-flash",
			name: "DeepSeek V4.1 Flash",
			api: "openai-responses",
			provider: "deepseek",
			compat: {
				supportsMidConvoSystemMessages: true,
				supportsToolSearch: true,
				requiresToolSearchTrailingNote: true,
			},
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string; description: string }>;
			input: Array<{ type?: string; role?: string; content?: string }>;
		}>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{
					role: "system",
					content: "",
					toolsAdded: [{ ...baseTool, description: "Updated contract" }],
					timestamp: 2,
				},
			],
		});
		expect(payload.tools?.map(({ name, description }) => ({ name, description }))).toEqual([
			{ name: "base_tool", description: "Updated contract" },
		]);
		expect(payload.input.some((item) => item.type === "tool_search_call")).toBe(false);
		expect(
			payload.input.some((item) => item.role === "developer" && item.content === TOOL_SEARCH_TRAILING_NOTE),
		).toBe(false);
	});

	test("anchors Kimi additions in tool-bearing system messages", async () => {
		const model: Model<"openai-completions"> = {
			...modelBase,
			id: "kimi-k3",
			name: "Kimi K3",
			api: "openai-completions",
			provider: "moonshotai",
			compat: { supportsMidConvoSystemMessages: true, supportsMidConvoToolAdditions: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ function?: { name: string } }>;
			messages: Array<{ role: string; content?: string; tools?: Array<{ function?: { name: string } }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool"]);
		expect(payload.messages.find((message) => message.tools)?.tools?.map((value) => value.function?.name)).toEqual([
			"late_tool",
		]);
		expect(payload.messages.filter((message) => message.role === "system").map((message) => message.content)).toEqual(
			["base prompt", undefined, "updated guidance"],
		);
	});

	type KimiPayload = {
		tools?: Array<{ function?: { name: string } }>;
		messages: Array<{ role: string; tools?: Array<{ function?: { name: string } }> }>;
	};

	// #10940: Kimi tool messages cannot remove tools, so a removal rewrites the prefix once.
	test("keeps the Kimi prefix across loads after a removal", async () => {
		const [beforeRemoval, ...afterRemoval] = await captureToolHistory<KimiPayload>(getModel("kimi-coding", "k3"));

		expect(beforeRemoval?.tools?.map((value) => value.function?.name)).toEqual(["base_tool", "drop_tool"]);
		for (const payload of afterRemoval) {
			expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool"]);
		}
		expectGrowingPrefix(afterRemoval.map((payload) => payload.messages));
		expect(
			afterRemoval
				.at(-1)
				?.messages.filter((message) => message.tools)
				.map((message) => message.tools?.[0]?.function?.name),
		).toEqual(["late_tool", "later_tool"]);
	});

	// #10720: Kimi Coding uses Chat Completions so K3-style models keep added tools out of the top-level list.
	test.each(["k3", "k3-256k", "kimi-for-coding"] as const)("anchors additions for Kimi Coding %s", async (modelId) => {
		const payload = await capturePayload<KimiPayload>(getModel("kimi-coding", modelId), additionContext);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool"]);
		expect(payload.messages.find((message) => message.tools)?.tools?.map((value) => value.function?.name)).toEqual([
			"late_tool",
		]);
	});

	// #10720: Kimi Code Highspeed rejects tool-bearing system messages.
	test("keeps Kimi Coding Highspeed additions in top-level tools", async () => {
		const payload = await capturePayload<KimiPayload>(
			getModel("kimi-coding", "kimi-for-coding-highspeed"),
			additionContext,
		);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.messages.some((message) => message.tools !== undefined)).toBe(false);
	});

	test("keeps Kimi K2 system text inline without dynamic tool messages", async () => {
		const model: Model<"openai-completions"> = {
			...modelBase,
			id: "kimi-k2.7-code",
			name: "Kimi K2.7 Code",
			api: "openai-completions",
			provider: "moonshotai",
			compat: { supportsMidConvoSystemMessages: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ function?: { name: string } }>;
			messages: Array<{ role: string; content?: string; tools?: Array<{ function?: { name: string } }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.messages.some((message) => message.tools !== undefined)).toBe(false);
		expect(payload.messages.filter((message) => message.role === "system").map((message) => message.content)).toEqual(
			["base prompt", "updated guidance"],
		);
	});

	test("folds OpenAI-compatible updates into the system prompt without native support", async () => {
		const model: Model<"openai-completions"> = {
			...modelBase,
			id: "custom-model",
			name: "Custom model",
			api: "openai-completions",
			provider: "custom-provider",
			reasoning: false,
		};
		const payload = await capturePayload<{
			tools?: Array<{ function?: { name: string } }>;
			messages: Array<{ role: string; content?: string }>;
		}>(model, context);

		expect(payload.tools?.map((value) => value.function?.name)).toEqual(["late_tool"]);
		expect(payload.messages.map((message) => message.role)).toEqual(["system", "user"]);
		expect(payload.messages[0]?.content).toBe("base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>");
	});
});

// A tool that leaves the transcript and returns unchanged is still in the flat field: the anchored path
// keeps the initial tools there and keeps a removed name in its slot. Re-declaring it as an addition made
// the request declare one name twice, which a strict endpoint rejects with "Tool names must be unique".
// The walk is key-agnostic because the items land under `input` on some transports and `messages` on others.
test("never declares one tool name twice across the flat field and the additions", async () => {
	const responses: Model<"openai-responses"> = {
		...modelBase,
		id: "gpt-5.4",
		name: "GPT-5.4",
		api: "openai-responses",
		provider: "openai",
		compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
	};

	const payload = await capturePayload<unknown>(responses, {
		messages: [
			{ role: "system", content: "base prompt", toolsAdded: [tool("base_tool"), tool("second_tool")], timestamp: 0 },
			{ role: "system", content: "", toolsRemoved: [{ name: "base_tool" }], timestamp: 1 },
			{ role: "system", content: "", toolsAdded: [tool("base_tool")], timestamp: 2 },
		],
	});

	const counts = new Map<string, number>();
	const visit = (node: unknown): void => {
		if (Array.isArray(node)) {
			for (const entry of node) visit(entry);
			return;
		}
		if (!node || typeof node !== "object") return;
		const record = node as Record<string, unknown>;
		if (Array.isArray(record.tools)) {
			for (const entry of record.tools) {
				const name = (entry as { name?: unknown }).name;
				if (typeof name === "string") counts.set(name, (counts.get(name) ?? 0) + 1);
			}
		}
		for (const value of Object.values(record)) visit(value);
	};
	visit(payload);

	expect([...counts].filter(([, count]) => count > 1)).toEqual([]);
	expect(counts.get("base_tool")).toBe(1);
	expect(counts.get("second_tool")).toBe(1);
});

// The shape that actually reached a provider: a later-added tool added again in a *later* message with no
// removal in between, as a goal transition does (update_goal added twice, removed zero times). Filtering
// only against the flat field does not catch it, because neither copy is in the initial tools.
test("never declares a tool twice when it is added again in a later message", async () => {
	const responses: Model<"openai-responses"> = {
		...modelBase,
		id: "gpt-5.4",
		name: "GPT-5.4",
		api: "openai-responses",
		provider: "openai",
		compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
	};

	const payload = await capturePayload<unknown>(responses, {
		messages: [
			{ role: "system", content: "base prompt", toolsAdded: [tool("base_tool")], timestamp: 0 },
			{
				role: "system",
				content: "goal created",
				toolsAdded: [tool("create_goal"), tool("update_goal")],
				timestamp: 1,
			},
			{ role: "system", content: "goal advanced", toolsAdded: [tool("update_goal")], timestamp: 2 },
		],
	});

	const counts = new Map<string, number>();
	const visit = (node: unknown): void => {
		if (Array.isArray(node)) {
			for (const entry of node) visit(entry);
			return;
		}
		if (!node || typeof node !== "object") return;
		const record = node as Record<string, unknown>;
		if (Array.isArray(record.tools)) {
			for (const entry of record.tools) {
				const name = (entry as { name?: unknown }).name;
				if (typeof name === "string") counts.set(name, (counts.get(name) ?? 0) + 1);
			}
		}
		for (const value of Object.values(record)) visit(value);
	};
	visit(payload);

	expect([...counts].filter(([, count]) => count > 1)).toEqual([]);
	expect(counts.get("update_goal")).toBe(1);
	expect(counts.get("create_goal")).toBe(1);
});

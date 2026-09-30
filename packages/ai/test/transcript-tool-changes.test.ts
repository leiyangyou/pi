import { Type } from "typebox";
import { describe, expect, test } from "vitest";
import { streamSimple } from "../src/compat.ts";
import type { Api, Context, Model, Tool } from "../src/types.ts";

class PayloadCaptured extends Error {}

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

interface AnthropicPayload {
	betas?: string[];
	system?: Array<{ text: string }>;
	tools?: Array<{ name: string; defer_loading?: boolean; cache_control?: unknown }>;
	messages: Array<{ role: string; content: Array<{ type: string; text?: string; tool?: { name: string } }> }>;
}

describe("transcript system messages", () => {
	test("sends Anthropic updates and tool changes in native system messages", async () => {
		const payload = await capturePayload<AnthropicPayload>(anthropicNativeModel, context);

		expect(payload.betas).toContain("mid-conversation-tool-changes-2026-07-01");
		expect(payload.system?.map((block) => block.text)).toEqual([
			"base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>",
		]);
		// Initial tools stay active and carry the cache breakpoint; the placeholder and every
		// later declaration are deferred; the removed tool stays declared.
		expect(payload.tools).toMatchObject([
			{ name: "base_tool", cache_control: { type: "ephemeral" } },
			{ name: "__pi_deferred_placeholder__", defer_loading: true },
			{ name: "late_tool", defer_loading: true },
		]);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
		expect(payload.tools?.[1]?.cache_control).toBeUndefined();
		expect(payload.tools?.[2]?.cache_control).toBeUndefined();
		const update = payload.messages.at(-1);
		expect(update).toMatchObject({
			role: "system",
			content: [
				{ type: "text" },
				{ type: "tool_removal", tool: { name: "base_tool" } },
				{ type: "tool_addition", tool: { name: "late_tool" } },
			],
		});
		expect(update?.content[0]?.text).toContain("updated guidance");
		expect(update?.content[0]?.text).toContain("<rules>\nnew rules\n</rules>");
		expect(update?.content[0]?.text).toContain('Removed system prompt section "docs"');

		// The placeholder is declared before any change so its scaffolding is cached from request one.
		const initial = await capturePayload<AnthropicPayload>(anthropicNativeModel, {
			messages: context.messages.slice(0, 2),
		});
		expect(initial.tools?.map((tool) => tool.name)).toEqual(["base_tool", "__pi_deferred_placeholder__"]);
	});

	test("sends the current Anthropic tool list when native tool changes cannot express the history", async () => {
		const redefinedTool = { ...baseTool, description: "changed" };
		const fallbackContexts: Context[] = [
			// Same-name redefinition: blocks reference tools by name only.
			{
				messages: [
					{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
					{
						role: "system",
						content: "updated guidance",
						toolsRemoved: [{ name: "base_tool" }],
						toolsAdded: [redefinedTool],
						timestamp: 2,
					},
				],
			},
			// No initial tool: Anthropic rejects an all-deferred tool list.
			{
				messages: [
					{ role: "system", content: "base prompt", timestamp: 0 },
					{ role: "system", content: "updated guidance", toolsAdded: [redefinedTool], timestamp: 2 },
				],
			},
		];
		for (const fallbackContext of fallbackContexts) {
			const payload = await capturePayload<AnthropicPayload>(anthropicNativeModel, fallbackContext);
			expect(payload.betas ?? []).not.toContain("mid-conversation-tool-changes-2026-07-01");
			expect(payload.tools).toMatchObject([
				{ name: "base_tool", description: "changed", cache_control: { type: "ephemeral" } },
			]);
			expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
			expect(payload.messages.at(-1)?.content.map((block) => block.type)).toEqual(["text"]);
		}
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

		expect(payload.betas ?? []).not.toContain("mid-conversation-tool-changes-2026-07-01");
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

		expect(payload.betas ?? []).not.toContain("mid-conversation-tool-changes-2026-07-01");
		expect(payload.tools?.map((value) => value.name)).toEqual(["late_tool"]);
		expect(payload.messages.map((message) => message.role)).toEqual(["user"]);
	});

	test("anchors OpenAI additions at their developer message", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; role?: string; content?: string; tools?: Array<{ name: string }> }>;
		}>(model, additionContext);

		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		expect(payload.input.find((item) => item.type === "additional_tools")?.tools?.map((value) => value.name)).toEqual(
			["late_tool"],
		);
		expect(
			payload.input
				.filter((item) => item.role === "developer" && item.type === undefined)
				.map((item) => item.content),
		).toEqual(["base prompt", "updated guidance"]);
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

	test("keeps a trailing note after anchored additions on every pair-bearing request", async () => {
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
		for (const payload of [current, followup]) {
			expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
			expect(payload.input.at(-1)).toEqual({
				role: "developer",
				content:
					"Tool definitions were updated above. Any tool listed in the latest tool search result is callable now.",
			});
			expect(
				payload.input.find((item) => item.type === "tool_search_output")?.tools?.map((tool) => tool.name),
			).toEqual(["late_tool"]);
		}
		const pair = (payload: Payload) => payload.input.findIndex((item) => item.type === "tool_search_call");
		expect(pair(current)).toBe(pair(followup));
		expect(current.input[pair(current)]?.call_id).toBe(followup.input[pair(followup)]?.call_id);
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

	// A removal is additive-safe: the removed tool keeps its slot in the flat field, which leaves it
	// declared, and the checkpoint's `tools` section patch is what tells the model it is gone. A call to
	// it is refused rather than executed, because the loadout no longer holds it. Only a redeclaration,
	// a name returning with a different interface, still replaces the flat field.
	test("keeps the flat tool field anchored when the change is a removal", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
		};
		const payload = await capturePayload<{
			tools?: Array<{ name: string }>;
			input: Array<{ type?: string; role?: string; tools?: Array<{ name: string }> }>;
		}>(model, context);

		// The initial declaration, unchanged, with the removed tool still in it.
		expect(payload.tools?.map((value) => value.name)).toEqual(["base_tool"]);
		// The later addition still rides its own in-place additions item.
		expect(payload.input.find((item) => item.type === "additional_tools")?.tools?.map((tool) => tool.name)).toEqual([
			"late_tool",
		]);
		// One developer item per in-place update: the folded guidance, the additions item, and the
		// removal's section patch.
		expect(payload.input.map((item) => item.type ?? item.role)).toEqual([
			"developer",
			"user",
			"additional_tools",
			"developer",
		]);
	});

	test("replaces the flat tool field when a name returns with a different interface", async () => {
		const model: Model<"openai-responses"> = {
			...modelBase,
			id: "gpt-5.4",
			name: "GPT-5.4",
			api: "openai-responses",
			provider: "openai",
			compat: { supportsMidConvoSystemMessages: true, supportsAdditionalTools: true },
		};
		const redefined = { ...baseTool, description: "a different interface" };
		const payload = await capturePayload<{ tools?: Array<{ name: string; description: string }> }>(model, {
			messages: [
				{ role: "system", content: "base prompt", toolsAdded: [baseTool], timestamp: 0 },
				{ role: "user", content: "before", timestamp: 1 },
				{
					role: "system",
					content: "",
					toolsRemoved: [{ name: baseTool.name }],
					toolsAdded: [redefined],
					timestamp: 2,
				},
			],
		});

		// A stale schema must not stay active, so the flat field carries the new interface.
		expect(payload.tools?.map((value) => value.name)).toEqual([baseTool.name]);
		expect(payload.tools?.[0]?.description).toBe("a different interface");
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
			payload.input.some(
				(item) => item.role === "developer" && item.content?.startsWith("Tool definitions were updated"),
			),
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

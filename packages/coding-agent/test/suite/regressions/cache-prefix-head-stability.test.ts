import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	fauxAssistantMessage,
	getCurrentSystemPrompt,
	getInitialSystemMessage,
	resolveTranscriptTools,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { fauxToolCall } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import type { InlineExtension } from "../../../src/index.ts";
import { createHarness, type Harness } from "../harness.ts";

/**
 * Prefix caching bills a cache read only for a common item prefix, so within one session the
 * request's `input` may only grow. Core owns the system checkpoints, so a `context` handler that
 * edits the conversation must not cost the head: the head carries the rendered prompt and the
 * initial tool declarations, and `getInitialSystemMessage` reads index 0 for both. Re-folding a
 * fresh head re-declares every live tool as the initial set, which materialises the request's flat
 * tool list and re-bills the whole conversation.
 */

function respondAndCapture(harness: Harness, text: string, captured: TranscriptContext[]): void {
	harness.setResponses([
		(context) => {
			captured.push(context);
			return fauxAssistantMessage(text);
		},
	]);
}

function firstDivergence(previous: readonly AgentMessage[], current: readonly AgentMessage[]) {
	for (let index = 0; index < previous.length; index++) {
		if (JSON.stringify(previous[index]) !== JSON.stringify(current[index])) {
			return { index, previous: previous[index]?.role, current: current[index]?.role };
		}
	}
	return undefined;
}

/** One turn that activates a snippet-bearing tool, as `create_goal` and an MCP connect both do. */
async function runToolActivationTurn(extensionFactories: InlineExtension[]) {
	const harness = await createHarness({ initialActiveToolNames: ["read"], extensionFactories });
	const captured: TranscriptContext[] = [];
	respondAndCapture(harness, "first answer", captured);
	await harness.session.prompt("first");
	harness.session.setActiveToolsByName(["read", "bash"]);
	respondAndCapture(harness, "second answer", captured);
	await harness.session.prompt("second");
	return { harness, previous: captured[0]?.messages, current: captured[1]?.messages };
}

/**
 * pi-goal-x records `goal_audit_event` entries into the transcript and then runs
 * `filterGoalSessionContext` on every request to drop them. That removal is the divergent
 * transcript the patch-independent route folds.
 */
const AUDIT_CUSTOM_TYPE = "goal_audit_event";

/** The runtime transcript carries custom messages that the `Message` type does not model. */
const isAuditEntry = (message: unknown): boolean =>
	(message as { customType?: string } | null)?.customType === AUDIT_CUSTOM_TYPE;

function countAuditEntries(messages: readonly unknown[]): number {
	return messages.filter(isAuditEntry).length;
}

/** The runtime transcript carries the head's `sections`; the `Message` type does not model them. */
function headToolsSection(messages: readonly unknown[] | undefined): string | undefined {
	const head = messages?.[0] as { sections?: Record<string, string> } | undefined;
	return head?.sections?.tools;
}

const dropAuditEntries: InlineExtension = (pi) => {
	let recorded = false;
	pi.on("agent_settled", () => {
		if (recorded) return;
		recorded = true;
		pi.sendMessage({ customType: AUDIT_CUSTOM_TYPE, content: "audit one", display: false }, { triggerTurn: false });
	});
	// pi-goal-x's `filterGoalSessionContext` reads `customType` off the message through the same
	// cast: the `context` event type does not surface custom messages, but they arrive at runtime.
	pi.on("context", async (event) => ({ messages: event.messages.filter((message) => !isAuditEntry(message)) }));
};

describe("head stability across a snippet-bearing tool activation", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("keeps the previous request as a prefix when no extension touches the conversation", async () => {
		const { harness, previous, current } = await runToolActivationTurn([]);
		harnesses.push(harness);
		expect(firstDivergence(previous ?? [], current ?? [])).toBeUndefined();
	});

	it("records the audit entry the handler then removes", async () => {
		const { harness, current } = await runToolActivationTurn([dropAuditEntries]);
		harnesses.push(harness);
		expect(countAuditEntries(harness.session.messages)).toBe(1);
		expect(countAuditEntries(current ?? [])).toBe(0);
	});

	it("keeps the previous request as a prefix when a handler removes a recorded item", async () => {
		const { harness, previous, current } = await runToolActivationTurn([dropAuditEntries]);
		harnesses.push(harness);
		expect(firstDivergence(previous ?? [], current ?? [])).toBeUndefined();
	});

	it("keeps the initial tool declaration when a handler removes a recorded item", async () => {
		const { harness, current } = await runToolActivationTurn([dropAuditEntries]);
		harnesses.push(harness);
		expect(getInitialSystemMessage(current ?? [])?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
	});

	// The flat tools field is what the provider caches alongside the head. Anchoring keeps it on
	// the initial declaration and lets the added definition ride the tool_search pair.
	it("keeps the request's flat tool list anchored when a handler removes a recorded item", async () => {
		const { harness, current } = await runToolActivationTurn([dropAuditEntries]);
		harnesses.push(harness);
		const tools = resolveTranscriptTools(current ?? [], true);
		expect(tools.anchorsAdditions).toBe(true);
		expect(tools.requestTools.map((tool) => tool.name)).toEqual(["read"]);
	});

	it("keeps the head at index 0 byte-identical when a handler inserts at the front", async () => {
		const block: AgentMessage = {
			role: "user",
			content: [{ type: "text", text: "<context_window>extension block</context_window>" }],
			timestamp: 0,
		};
		let turn = 0;
		const { harness, previous, current } = await runToolActivationTurn([
			(pi) => {
				pi.on("context", async (event) => {
					if (++turn !== 2) return;
					return { messages: [block, ...event.messages] };
				});
			},
		]);
		harnesses.push(harness);
		expect(current?.[0]?.role).toBe("system");
		expect(JSON.stringify(current?.[0])).toBe(JSON.stringify(previous?.[0]));
	});
});

/** The activation the live trigger performs, as `create_goal` and an MCP connect both do. */
const activateOnSecondTurn: InlineExtension = (pi) => {
	let turn = 0;
	pi.on("before_agent_start", () => {
		if (++turn === 2) pi.setActiveTools(["read", "bash"]);
	});
};

/**
 * context-mode's own carrier. `build/adapters/pi/extension.js:565-566` reads `event.systemPrompt`
 * into `parts`, and its `context` handler appends the joined result as a user message. So the live
 * rendered prompt is copied into a conversation item, and whatever moves the render moves that item.
 */
const embedLivePrompt: InlineExtension = (pi) => {
	let pending: string | undefined;
	pi.on("before_agent_start", (event) => {
		pending = String(event.systemPrompt ?? "");
	});
	pi.on("context", (event) => {
		if (pending === undefined) return;
		event.messages.push({ role: "user", content: pending, timestamp: 0 });
		pending = undefined;
		return { messages: event.messages };
	});
};

/** Two turns, where the second turn's `before_agent_start` grows the active tool set. */
async function runActivationTurn(extensionFactories: InlineExtension[]) {
	const harness = await createHarness({ initialActiveToolNames: ["read"], extensionFactories });
	const captured: TranscriptContext[] = [];
	respondAndCapture(harness, "first answer", captured);
	await harness.session.prompt("first");
	respondAndCapture(harness, "second answer", captured);
	await harness.session.prompt("second");
	return { harness, previous: captured[0]?.messages, current: captured[1]?.messages };
}

describe("head stability when a handler activates a tool at a turn boundary", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("activates the tool", async () => {
		const { harness } = await runActivationTurn([activateOnSecondTurn]);
		harnesses.push(harness);
		expect(harness.session.getActiveToolNames()).toEqual(["read", "bash"]);
	});

	// A pure addition is not a head change. The session keeps the index it rendered for the run and
	// declares the addition in a checkpoint, which is what lets `resolveTranscriptTools` anchor it.
	it("leaves the rendered tool index alone when the activation is an addition", async () => {
		const { harness, previous, current } = await runActivationTurn([activateOnSecondTurn]);
		harnesses.push(harness);
		expect(headToolsSection(current)).toBe(headToolsSection(previous));
		expect(headToolsSection(current)).toContain("- read:");
	});

	it("keeps the previous request as a prefix of the next one", async () => {
		const { harness, previous, current } = await runActivationTurn([activateOnSecondTurn]);
		harnesses.push(harness);
		expect(firstDivergence(previous ?? [], current ?? [])).toBeUndefined();
	});
});

describe("head stability when a carrier copies the live rendered prompt into the conversation", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	// The carrier's copy is appended at its own position, so the divergence it causes is at that
	// position, not at the head. Turn one sends the block before the assistant reply, which the next
	// request cannot reproduce, so the prefix ends there. That is the carrier's placement choice and
	// it is near the tail; the head is what has to stay put, and it does.
	it("leaves the head at index 0 byte-identical", async () => {
		const { harness, previous, current } = await runActivationTurn([activateOnSecondTurn, embedLivePrompt]);
		harnesses.push(harness);
		expect(current?.[0]?.role).toBe("system");
		expect(JSON.stringify(current?.[0])).toBe(JSON.stringify(previous?.[0]));
		expect(headToolsSection(current)).toBe(headToolsSection(previous));
	});
});

/**
 * The live trigger's other shape: the activation happens inside the agent run, after the first
 * provider request, as `create_goal` and an MCP connect both do. The head stays put here too.
 */
async function runInsideRunActivation() {
	const harness = await createHarness({
		initialActiveToolNames: ["read"],
		extensionFactories: [
			(pi) => {
				pi.on("tool_execution_end", () => {
					pi.setActiveTools(["read", "bash"]);
				});
			},
		],
	});
	const captured: TranscriptContext[] = [];
	harness.setResponses([
		(context) => {
			captured.push(context);
			return fauxAssistantMessage([fauxToolCall("read", { path: "missing.txt" })]);
		},
		(context) => {
			captured.push(context);
			return fauxAssistantMessage("second answer");
		},
	]);
	await harness.session.prompt("go");
	return { harness, previous: captured[0]?.messages, current: captured[1]?.messages };
}

describe("head stability when a tool is activated inside one agent run", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("activates the tool between the run's two requests", async () => {
		const { harness, previous, current } = await runInsideRunActivation();
		harnesses.push(harness);
		expect(previous).toBeDefined();
		expect(current).toBeDefined();
		expect(harness.session.getActiveToolNames()).toEqual(["read", "bash"]);
	});

	it("keeps the previous request as a prefix of the next one", async () => {
		const { harness, previous, current } = await runInsideRunActivation();
		harnesses.push(harness);
		expect(firstDivergence(previous ?? [], current ?? [])).toBeUndefined();
	});
});

/**
 * A section-only change, which is what a ponytail mode switch and a subagent advertisement change
 * both are. The section patch is conceptually an append: `getCurrentSystemMessage` merges `sections`
 * in order, so a later message wins wherever it sits.
 */
describe("head stability across a section-only change", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("reports what a section change does to the leading item", async () => {
		let turn = 0;
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			extensionFactories: [
				(pi) => {
					pi.on("before_agent_start", (event) => {
						if (++turn !== 2) return;
						const options = event.systemPromptOptions as { sections?: Record<string, string> };
						if (options.sections) options.sections.plan_mode = "Plan only.";
					});
				},
			],
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "first answer", captured);
		await harness.session.prompt("first");
		respondAndCapture(harness, "second answer", captured);
		await harness.session.prompt("second");

		const previous = captured[0]?.messages ?? [];
		const current = captured[1]?.messages ?? [];
		expect(current[0]?.role).toBe("system");
		expect(getCurrentSystemPrompt(current)).toContain("Plan only.");
		// Mechanism (b): `normalizeContext` prepends a head built from the live `context.systemPrompt`,
		// so a section change would rewrite item 0. If this holds, the change is an append instead.
		expect(JSON.stringify(current[0])).toBe(JSON.stringify(previous[0]));
		// Mechanism (a): `messages.unshift(updateMessage)` in `_runAgentPrompt`. If the patch lands
		// ahead of what the previous request already sent, the prefix breaks where it lands.
		expect(firstDivergence(previous, current)).toBeUndefined();
	});
});

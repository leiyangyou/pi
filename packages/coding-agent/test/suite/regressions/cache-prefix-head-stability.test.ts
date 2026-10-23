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
import { type SessionEntry, SessionManager } from "../../../src/core/session-manager.ts";
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
		const tools = resolveTranscriptTools(current ?? [], "additions");
		expect(tools.foldedSystemMessages).toBe(1);
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

/**
 * A carrier that records its block at session start, as pi-context's `<context_window>` boot does.
 * `convertToLlm` sends a custom message as a `user` item, so the recorded block precedes the head in
 * every request of the session. Seeded on the session manager because a `session_start` handler's
 * `sendMessage` does not reach the recorded transcript in the harness.
 */
const CARRIER_CUSTOM_TYPE = "carrier/boot";

function seedCarrierBlock(): SessionManager {
	const sessionManager = SessionManager.inMemory();
	sessionManager.appendCustomMessageEntry(
		CARRIER_CUSTOM_TYPE,
		"<context_window>carrier block</context_window>",
		false,
	);
	return sessionManager;
}

/** Two turns where the second activates a snippet-bearing tool, on a session whose transcript leads
 * with a carrier block. */
async function runToolActivationTurnWithCarrier() {
	const harness = await createHarness({
		initialActiveToolNames: ["read"],
		sessionManager: seedCarrierBlock(),
	});
	const captured: TranscriptContext[] = [];
	respondAndCapture(harness, "first answer", captured);
	await harness.session.prompt("first");
	harness.session.setActiveToolsByName(["read", "bash"]);
	respondAndCapture(harness, "second answer", captured);
	await harness.session.prompt("second");
	return { harness, previous: captured[0]?.messages, current: captured[1]?.messages };
}

/**
 * A transcript recorded before the head was marked, with a shell result ahead of it — the shape a session
 * that runs `!` before its first prompt leaves behind. Pi writes its head before its first turn, so the
 * head is still recoverable from provenance when no marker was recorded.
 */
function seedLegacyShellFirstSession(): SessionManager {
	const stamp = Date.now();
	const entry = (id: string, parentId: string | null, offset: number, message: AgentMessage): SessionEntry => ({
		type: "message",
		id,
		parentId,
		timestamp: new Date(stamp + offset).toISOString(),
		message,
	});
	return SessionManager.inMemory(undefined, undefined, [
		entry("legacy-bash", null, 0, {
			role: "bashExecution",
			command: "printf boot",
			output: "boot",
			exitCode: 0,
			cancelled: false,
			truncated: false,
			timestamp: stamp,
		}),
		entry("legacy-head", "legacy-bash", 1, {
			role: "system",
			content: "",
			sections: {
				preamble: "<preamble>legacy prompt</preamble>",
				tools: "<tools>\n- read: Read a file.\n</tools>",
				rules: "<rules>\n- legacy rule\n</rules>",
			},
			toolsAdded: [
				{
					name: "read",
					description: "Read a file.",
					parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
				},
			],
			timestamp: stamp + 1,
		}),
		entry("legacy-user", "legacy-head", 2, { role: "user", content: "go", timestamp: stamp + 2 }),
	]);
}

/**
 * A legacy compaction that stored no head, with its retained turn then edited away and an unmarked
 * post-compaction delta behind it. Promoting that delta would make its partial sections the request's base
 * prompt, so recovery has to stop at the compaction summary.
 */
function seedLegacyPostCompactionDelta(): SessionManager {
	const stamp = Date.now();
	const entry = (id: string, parentId: string | null, offset: number, message: AgentMessage): SessionEntry => ({
		type: "message",
		id,
		parentId,
		timestamp: new Date(stamp + offset).toISOString(),
		message,
	});
	const manager = SessionManager.inMemory(undefined, undefined, [
		{
			type: "compaction",
			id: "legacy-compaction",
			parentId: null,
			timestamp: new Date(stamp).toISOString(),
			summary: "Earlier work summarized.",
			firstKeptEntryId: "legacy-turn",
			tokensBefore: 1000,
		},
		entry("legacy-turn", "legacy-compaction", 1, {
			role: "user",
			content: "retained turn",
			timestamp: stamp + 1,
		}),
		entry("legacy-delta", "legacy-turn", 2, {
			role: "system",
			content: "",
			sections: { rules: "<rules>\n- delta rule\n</rules>" },
			timestamp: stamp + 2,
		}),
	]);
	manager.appendContextEdit("legacy-turn", null);
	return manager;
}

/**
 * A legacy compaction that stored no head, followed by the complete head a later prompt wrote before markers
 * existed. Recovery must lead with that head — refusing every system message behind a summary would strand
 * it — while a delta behind a turn stays where it is.
 */
function seedLegacyCompactionHead(): SessionManager {
	const stamp = Date.now();
	const entry = (id: string, parentId: string | null, offset: number, message: AgentMessage): SessionEntry => ({
		type: "message",
		id,
		parentId,
		timestamp: new Date(stamp + offset).toISOString(),
		message,
	});
	return SessionManager.inMemory(undefined, undefined, [
		{
			type: "compaction",
			id: "legacy-compaction",
			parentId: null,
			timestamp: new Date(stamp).toISOString(),
			summary: "Earlier work summarized.",
			firstKeptEntryId: "legacy-head",
			tokensBefore: 1000,
		},
		entry("legacy-head", "legacy-compaction", 1, {
			role: "system",
			content: "",
			sections: {
				preamble: "<preamble>legacy prompt</preamble>",
				tools: "<tools>\n- read: Read a file.\n</tools>",
				rules: "<rules>\n- legacy rule\n</rules>",
			},
			toolsAdded: [
				{
					name: "read",
					description: "Read a file.",
					parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
				},
			],
			timestamp: stamp + 1,
		}),
		entry("legacy-user", "legacy-head", 2, { role: "user", content: "go", timestamp: stamp + 2 }),
	]);
}

/**
 * A checkpoint-less legacy compaction that retained an earlier turn, followed by the complete head the next
 * prompt wrote. The retained turn sits ahead of the head in the projection but predates the compaction, so it
 * says nothing about the region and must not disqualify the head.
 */
function seedLegacyCompactionWithRetainedTurn(): SessionManager {
	const stamp = Date.now();
	const entry = (id: string, parentId: string | null, offset: number, message: AgentMessage): SessionEntry => ({
		type: "message",
		id,
		parentId,
		timestamp: new Date(stamp + offset).toISOString(),
		message,
	});
	return SessionManager.inMemory(undefined, undefined, [
		entry("legacy-old-user", null, 0, { role: "user", content: "older turn", timestamp: stamp }),
		{
			type: "compaction",
			id: "legacy-compaction",
			parentId: "legacy-old-user",
			timestamp: new Date(stamp + 1).toISOString(),
			summary: "Earlier work summarized.",
			firstKeptEntryId: "legacy-old-user",
			tokensBefore: 1000,
		},
		entry("legacy-head", "legacy-compaction", 2, {
			role: "system",
			content: "",
			sections: {
				preamble: "<preamble>legacy prompt</preamble>",
				tools: "<tools>\n- read: Read a file.\n</tools>",
				rules: "<rules>\n- legacy rule\n</rules>",
			},
			toolsAdded: [
				{
					name: "read",
					description: "Read a file.",
					parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
				},
			],
			timestamp: stamp + 2,
		}),
		entry("legacy-new-user", "legacy-head", 3, { role: "user", content: "go", timestamp: stamp + 3 }),
	]);
}

/**
 * The head carries the prompt and the initial tool declarations, and a provider reads both from the
 * leading item of the request. A carrier's recorded block must therefore not lead: with it at index 0,
 * `getInitialSystemMessage` returns undefined, the flat tool list is materialised from every live tool,
 * and each activation rewrites the front of the request and re-bills the conversation behind it.
 */
describe("head stability when a carrier block precedes the head in the transcript", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("records the carrier block ahead of the head", async () => {
		const { harness } = await runToolActivationTurnWithCarrier();
		harnesses.push(harness);
		const recorded = harness.sessionManager.buildSessionContext().messages as Array<{
			role: string;
			customType?: string;
		}>;
		const carrierAt = recorded.findIndex((message) => message.customType === CARRIER_CUSTOM_TYPE);
		const headAt = recorded.findIndex((message) => message.role === "system");
		expect(carrierAt).toBeGreaterThanOrEqual(0);
		expect(headAt).toBeGreaterThan(carrierAt);
	});

	it("leads the request with the head that declares the initial tools", async () => {
		const { harness, current } = await runToolActivationTurnWithCarrier();
		harnesses.push(harness);
		expect(current?.[0]?.role).toBe("system");
		expect(getInitialSystemMessage(current ?? [])?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
	});

	it("keeps the request's flat tool list anchored across the activation", async () => {
		const { harness, current } = await runToolActivationTurnWithCarrier();
		harnesses.push(harness);
		const tools = resolveTranscriptTools(current ?? [], "additions");
		expect(tools.foldedSystemMessages).toBe(1);
		expect(tools.requestTools.map((tool) => tool.name)).toEqual(["read"]);
	});

	it("keeps the previous request as a prefix of the next one", async () => {
		const { harness, previous, current } = await runToolActivationTurnWithCarrier();
		harnesses.push(harness);
		expect(firstDivergence(previous ?? [], current ?? [])).toBeUndefined();
	});

	// The head is identified by provenance, not by what it declares: a session that starts with no
	// tools has a head without `toolsAdded`, and a later checkpoint declares the first one. Reading the
	// head as "the first system message declaring tools" would promote that delta instead, losing the
	// preamble, docs and cwd from the provider's leading prompt.
	it("leads with a tool-less head rather than a later tool checkpoint", async () => {
		const harness = await createHarness({ initialActiveToolNames: [], sessionManager: seedCarrierBlock() });
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "first answer", captured);
		await harness.session.prompt("first");
		harness.session.setActiveToolsByName(["read"]);
		respondAndCapture(harness, "second answer", captured);
		await harness.session.prompt("second");

		const recorded = harness.sessionManager.buildSessionContext().messages as Array<{
			role: string;
			timestamp?: number;
		}>;
		const headTimestamp = recorded.find((message) => message.role === "system")?.timestamp;
		const first = (captured[0]?.messages ?? []) as Array<{ role: string; timestamp?: number }>;
		const second = (captured[1]?.messages ?? []) as Array<{ role: string; timestamp?: number }>;
		expect(first[0]?.role).toBe("system");
		expect(first[0]?.timestamp).toBe(headTimestamp);
		// The head is the message that renders no tools, because the session declared none at that point;
		// a promoted delta checkpoint would render the tool it added instead.
		expect(headToolsSection(first)).toContain("(none)");
		// The delta the activation records stays behind the head, where it belongs.
		expect(second.filter((message) => message.role === "system").map((message) => message.timestamp)).toEqual([
			headTimestamp,
			expect.any(Number),
		]);
	});

	// `context_with_system` handlers are documented to own the prompt: their output is sent as
	// returned. Pi leads with the head before that phase, so it never overrides the owner.
	it("sends what a context_with_system handler returns, in its own order", async () => {
		const preamble: AgentMessage = { role: "system", content: "Handler owns the prompt.", timestamp: 1 };
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			sessionManager: seedCarrierBlock(),
			extensionFactories: [
				(pi) => {
					pi.on("context_with_system", (event) => ({ messages: [preamble, ...event.messages] }));
				},
			],
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "first answer", captured);
		await harness.session.prompt("first");
		expect(JSON.stringify(captured[0]?.messages?.[0])).toBe(JSON.stringify(preamble));
	});

	// A shell result can be recorded before the session writes its head, so the head is not the first
	// item and no role-based rule can recognise it. The head is marked when it is written instead.
	it("leads with the head when a shell result was recorded before it", async () => {
		const harness = await createHarness({ initialActiveToolNames: ["read"] });
		harnesses.push(harness);
		harness.session.recordBashResult("printf boot", {
			output: "boot",
			exitCode: 0,
			cancelled: false,
			truncated: false,
		});
		harness.session.setActiveToolsByName(["read"]);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "first answer", captured);
		await harness.session.prompt("first");
		harness.session.setActiveToolsByName(["read", "bash"]);
		respondAndCapture(harness, "second answer", captured);
		await harness.session.prompt("second");

		const first = captured[0]?.messages ?? [];
		const second = captured[1]?.messages ?? [];
		expect(first[0]?.role).toBe("system");
		expect(JSON.stringify(second[0])).toBe(JSON.stringify(first[0]));
		expect(resolveTranscriptTools(second, "additions").requestTools.map((tool) => tool.name)).toEqual(["read"]);
	});

	// A branch summary can open a new root, where the next prompt writes a fresh head behind it.
	it("leads with a head written after a branch summary opened a new root", async () => {
		const harness = await createHarness({ initialActiveToolNames: ["read"] });
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "first answer", captured);
		await harness.session.prompt("first");
		harness.sessionManager.branchWithSummary(null, "Earlier work summarized.");
		// The product refreshes the finalized context after a tree change; the next prompt writes a fresh head.
		harness.session.refreshContext();
		respondAndCapture(harness, "second answer", captured);
		await harness.session.prompt("again");

		const current = captured[1]?.messages ?? [];
		expect(current[0]?.role).toBe("system");
		expect(getInitialSystemMessage(current)?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
	});

	// A transcript recorded before the head was marked carries no identity, so the head is recovered from
	// how a transcript is built: only entries recorded before the first turn may precede it.
	it("leads with a legacy head that a shell result precedes", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			sessionManager: seedLegacyShellFirstSession(),
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "answer", captured);
		await harness.session.prompt("second");

		const current = captured[0]?.messages ?? [];
		expect(current[0]?.role).toBe("system");
		expect(getInitialSystemMessage(current)?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
		// The recovered head anchors the flat tool list, and the seeded region keeps its head unmarked.
		expect(resolveTranscriptTools(current, "additions").requestTools.map((tool) => tool.name)).toEqual(["read"]);
		const recorded = harness.sessionManager.buildSessionContext().messages as Array<{
			role: string;
			leading?: true;
		}>;
		expect(recorded.some((message) => message.leading === true)).toBe(false);
	});

	// A legacy compaction that stored no head leaves a delta as the earliest system message. Promoting it
	// would make a partial prompt and tool delta the request's base, so recovery stops at the summary and
	// the transcript keeps the behaviour it had before this change.
	it("leaves a legacy transcript without a head alone rather than promoting a delta", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			sessionManager: seedLegacyPostCompactionDelta(),
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "answer", captured);
		await harness.session.prompt("second");

		const current = captured[0]?.messages ?? [];
		expect(getInitialSystemMessage(current)).toBeUndefined();
		// The delta is still there, in place, behind the summary.
		expect(current.some((message) => message.role === "system")).toBe(true);
	});

	// The other side of that boundary: a checkpoint-less legacy compaction whose later prompt wrote a
	// complete head. That head is not a delta, so recovery has to lead with it.
	it("leads with a complete unmarked head behind a checkpoint-less legacy compaction", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			sessionManager: seedLegacyCompactionHead(),
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "answer", captured);
		await harness.session.prompt("second");

		const current = captured[0]?.messages ?? [];
		expect(current[0]?.role).toBe("system");
		expect(getInitialSystemMessage(current)?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
		expect(resolveTranscriptTools(current, "additions").requestTools.map((tool) => tool.name)).toEqual(["read"]);
	});

	// A compaction re-inserts the turns it retained ahead of the region's own entries, so a retained turn
	// predates the region and must not disqualify the head the region wrote.
	it("leads with an unmarked head behind turns a legacy compaction retained", async () => {
		const harness = await createHarness({
			initialActiveToolNames: ["read"],
			sessionManager: seedLegacyCompactionWithRetainedTurn(),
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		respondAndCapture(harness, "answer", captured);
		await harness.session.prompt("second");

		const current = captured[0]?.messages ?? [];
		expect(current[0]?.role).toBe("system");
		expect(getInitialSystemMessage(current)?.toolsAdded?.map((tool) => tool.name)).toEqual(["read"]);
		expect(resolveTranscriptTools(current, "additions").requestTools.map((tool) => tool.name)).toEqual(["read"]);
	});
});

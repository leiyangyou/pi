import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	fauxAssistantMessage,
	getInitialSystemMessage,
	resolveTranscriptTools,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
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

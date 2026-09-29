import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, resolveTranscriptTools, type TranscriptContext } from "@earendil-works/pi-ai";
import { fauxToolCall } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, getMessageText, type Harness } from "../harness.ts";

/**
 * Removing a tool from the loadout no longer rewrites the request's flat tool field, because a rewrite
 * there costs the whole conversation's cache prefix. The tool therefore stays declared, and the safety
 * half of that trade is that a call to it must be refused rather than executed: `prepareToolCall`
 * resolves against `currentContext.tools` and returns `Tool <name> not found` when the loadout has
 * dropped it (`packages/agent/src/agent-loop.ts:710-716`).
 *
 * These two assertions are a pair. A reader who removes the refusal must fail this file, and a reader who
 * reinstates the flat-field rewrite must fail it too.
 */

function recordingTool(name: string, ran: string[]): AgentTool {
	return {
		name,
		label: name,
		description: `the ${name} tool`,
		parameters: Type.Object({}),
		execute: async () => {
			ran.push(name);
			return { content: [{ type: "text" as const, text: `${name} ran` }], details: {} };
		},
	};
}

describe("a tool removed from the loadout", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("stays declared in the flat tool field and is refused when the model calls it", async () => {
		const ran: string[] = [];
		const harness = await createHarness({
			tools: [recordingTool("alpha", ran), recordingTool("beta", ran)],
			initialActiveToolNames: ["alpha", "beta"],
		});
		harnesses.push(harness);
		const captured: TranscriptContext[] = [];
		harness.setResponses([
			(context) => {
				captured.push(context);
				return fauxAssistantMessage("first");
			},
			(context) => {
				captured.push(context);
				return fauxAssistantMessage([fauxToolCall("beta", {})]);
			},
			(context) => {
				captured.push(context);
				return fauxAssistantMessage("done");
			},
		]);

		await harness.session.prompt("first");
		harness.session.setActiveToolsByName(["alpha"]);
		await harness.session.prompt("second");

		// The declaration survived, so the cached prefix does.
		const declared = resolveTranscriptTools(captured[1]?.messages ?? [], true);
		expect(declared.anchorsAdditions).toBe(true);
		expect(declared.requestTools.map((tool) => tool.name).sort()).toEqual(["alpha", "beta"]);

		// And the tool did not run.
		expect(ran).toEqual([]);
		const refusals = harness.session.messages
			.filter((message) => message.role === "toolResult")
			.map((message) => getMessageText(message));
		expect(refusals).toEqual([expect.stringContaining("Tool beta not found")]);
	});
});

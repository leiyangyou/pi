import { fauxAssistantMessage, type TranscriptContext } from "@earendil-works/pi-ai";
import { expect, it } from "vitest";
import { createHarness } from "../harness.ts";

it("keeps tool additions in the provider transcript when the prompt is forced", async () => {
	const harness = await createHarness({
		initialActiveToolNames: ["read"],
		extensionFactories: [
			(pi) => {
				pi.on("before_agent_start", (event) => ({ systemPrompt: `${event.systemPrompt}\nPolicy.` }));
			},
		],
	});
	try {
		harness.setResponses([fauxAssistantMessage("first")]);
		await harness.session.prompt("first");
		harness.session.setActiveToolsByName(["read", "bash"]);
		let transcript: TranscriptContext | undefined;
		harness.setResponses([
			(context) => {
				transcript = context;
				return fauxAssistantMessage("second");
			},
		]);
		await harness.session.prompt("second");
		if (!transcript) throw new Error("No provider request");
		const systems = transcript.messages.filter((message) => message.role === "system");
		expect(systems.map((message) => message.toolsAdded?.map((tool) => tool.name) ?? [])).toEqual([
			["read"],
			["bash"],
		]);
		expect(systems[0]?.content).toContain("Policy.");
		expect(systems[1]?.content).toBe("");
		expect(systems.every((message) => !("sections" in message))).toBe(true);
	} finally {
		harness.cleanup();
	}
});

it("retains a permitted tool removal while masking the old prompt", async () => {
	let turn = 0;
	const harness = await createHarness({
		initialActiveToolNames: ["read", "bash"],
		extensionFactories: [
			(pi) => {
				pi.on("before_agent_start", (event) => {
					if (++turn > 1) event.systemPromptOptions.selectedTools = ["read"];
					return { systemPrompt: "Permission: only read." };
				});
			},
		],
	});
	try {
		harness.setResponses([fauxAssistantMessage("first")]);
		await harness.session.prompt("first");
		let transcript: TranscriptContext | undefined;
		harness.setResponses([
			(context) => {
				transcript = context;
				return fauxAssistantMessage("second");
			},
		]);
		await harness.session.prompt("second");
		if (!transcript) throw new Error("No provider request");
		const systems = transcript.messages.filter((message) => message.role === "system");
		expect(systems[0]?.content).toBe("Permission: only read.");
		expect(systems[0]?.toolsAdded?.map((tool) => tool.name)).toEqual(["read", "bash"]);
		expect(systems.at(-1)?.toolsRemoved?.map((tool) => tool.name)).toEqual(["bash"]);
		expect(systems.at(-1)?.content).toBe("");
		expect(harness.session.getActiveToolNames()).toEqual(["read"]);
	} finally {
		harness.cleanup();
	}
});

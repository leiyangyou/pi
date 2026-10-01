import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import type { ExtensionAPI, InputEvent } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ProcessImageResult } from "../../src/utils/image-process.ts";
import { createHarness, getAssistantTexts, getMessageText, getUserTexts, type Harness } from "./harness.ts";

const processImage = vi.hoisted(() =>
	vi.fn(
		async (_bytes: Uint8Array, mimeType: string): Promise<ProcessImageResult> => ({
			ok: true,
			data: Buffer.from("normalized").toString("base64"),
			mimeType,
			hints: [],
		}),
	),
);
vi.mock("../../src/utils/image-process.ts", () => ({ processImage }));

const DIMENSION_HINT =
	"[Image: original 2156x74, displayed at 2000x69. Multiply coordinates by 1.08 to map to original image.]";
const OVERSIZED_IMAGE = { type: "image" as const, mimeType: "image/png", data: Buffer.from("raw").toString("base64") };

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve = () => {};
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

async function createWaitingHarness(
	options: {
		tools?: AgentTool[];
		extensionFactories?: Harness["session"]["extensionRunner"] extends never
			? never
			: Array<(pi: ExtensionAPI) => void>;
	} = {},
): Promise<{
	harness: Harness;
	releaseToolExecution: () => void;
	promptPromise: Promise<void>;
	waitForToolStart: Promise<void>;
}> {
	let releaseToolExecution: (() => void) | undefined;
	const toolRelease = new Promise<void>((resolve) => {
		releaseToolExecution = resolve;
	});
	const waitTool: AgentTool = {
		name: "wait",
		label: "Wait",
		description: "Wait for release",
		parameters: Type.Object({}),
		execute: async () => {
			await toolRelease;
			return {
				content: [{ type: "text", text: "released" }],
				details: {},
			};
		},
	};
	const harness = await createHarness({
		tools: [waitTool, ...(options.tools ?? [])],
		extensionFactories: options.extensionFactories,
	});

	const waitForToolStart = new Promise<void>((resolve) => {
		const unsubscribe = harness.session.subscribe((event) => {
			if (event.type === "tool_execution_start" && event.toolName === "wait") {
				unsubscribe();
				resolve();
			}
		});
	});

	return {
		harness,
		releaseToolExecution: () => releaseToolExecution?.(),
		promptPromise: harness.session.prompt("start"),
		waitForToolStart,
	};
}

describe("AgentSession queue characterization", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		processImage.mockClear();
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("dispatches extension commands immediately when prompted while idle", async () => {
		const commandRuns: string[] = [];
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async (args) => {
							commandRuns.push(args);
						},
					});
				},
			],
		});
		harnesses.push(harness);

		await harness.session.prompt("/testcmd hello world");

		expect(commandRuns).toEqual(["hello world"]);
		expect(harness.getPendingResponseCount()).toBe(0);
		expect(harness.session.messages).toEqual([]);
	});

	it("delivers extension-origin steering messages before the next LLM call", async () => {
		let extensionApi: ExtensionAPI | undefined;
		const waiting = await createWaitingHarness({
			extensionFactories: [
				(pi) => {
					extensionApi = pi;
				},
			],
		});
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				const sawSteer = context.messages.some(
					(message) => message.role === "user" && getMessageText(message) === "steer now",
				);
				return fauxAssistantMessage(sawSteer ? "saw steer" : "missing steer");
			},
		]);

		await waitForToolStart;
		await new Promise((resolve) => setTimeout(resolve, 0));

		extensionApi?.sendUserMessage("steer now", { deliverAs: "steer" });
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "steer now"]);
		expect(getAssistantTexts(harness)).toContain("saw steer");
	});

	it("delivers follow-up messages only after the current run finishes", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const assistantSeenBeforeFollowUp: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				assistantSeenBeforeFollowUp.push(
					...context.messages
						.filter((message) => message.role === "assistant")
						.map((message) =>
							message.content
								.filter((part): part is { type: "text"; text: string } => part.type === "text")
								.map((part) => part.text)
								.join("\n"),
						),
				);
				return fauxAssistantMessage("follow-up response");
			},
		]);

		await waitForToolStart;
		await harness.session.followUp("after current run");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "after current run"]);
		expect(assistantSeenBeforeFollowUp).toContain("");
		expect(getAssistantTexts(harness)).toContain("follow-up response");
	});

	// Regression test for #8718.
	it("runs direct steering and follow-up messages through input handlers", async () => {
		const inputEvents: Array<Pick<InputEvent, "text" | "source" | "streamingBehavior">> = [];
		const waiting = await createWaitingHarness({
			extensionFactories: [
				(pi) => {
					pi.on("input", (event) => {
						inputEvents.push({
							text: event.text,
							source: event.source,
							streamingBehavior: event.streamingBehavior,
						});
						if (event.text.startsWith("handle")) return { action: "handled" };
						return { action: "transform", text: `transformed: ${event.text}` };
					});
				},
			],
		});
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("steered"),
			fauxAssistantMessage("followed up"),
		]);

		await waitForToolStart;
		inputEvents.length = 0;
		try {
			await harness.session.steer("steer me", undefined, { source: "rpc" });
			await harness.session.steer("handle steer", undefined, { source: "rpc" });
			await harness.session.followUp("follow me", undefined, { source: "rpc" });
			await harness.session.followUp("handle follow", undefined, { source: "rpc" });

			expect(inputEvents).toEqual([
				{ text: "steer me", source: "rpc", streamingBehavior: "steer" },
				{ text: "handle steer", source: "rpc", streamingBehavior: "steer" },
				{ text: "follow me", source: "rpc", streamingBehavior: "followUp" },
				{ text: "handle follow", source: "rpc", streamingBehavior: "followUp" },
			]);
			expect(harness.session.getSteeringMessages()).toEqual(["transformed: steer me"]);
			expect(harness.session.getFollowUpMessages()).toEqual(["transformed: follow me"]);
		} finally {
			releaseToolExecution();
		}
		await promptPromise;
	});

	it("delivers multiple steering messages in order in one-at-a-time mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("handled steer 1"),
			fauxAssistantMessage("handled steer 2"),
		]);

		await waitForToolStart;
		await harness.session.steer("steer 1");
		await harness.session.steer("steer 2");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "steer 1", "steer 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "handled steer 1", "handled steer 2"]);
	});

	it("delivers multiple follow-up messages in order in one-at-a-time mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			fauxAssistantMessage("handled follow-up 1"),
			fauxAssistantMessage("handled follow-up 2"),
		]);

		await waitForToolStart;
		await harness.session.followUp("follow-up 1");
		await harness.session.followUp("follow-up 2");
		releaseToolExecution();
		await promptPromise;

		expect(getUserTexts(harness)).toEqual(["start", "follow-up 1", "follow-up 2"]);
		expect(getAssistantTexts(harness)).toEqual([
			"",
			"original turn complete",
			"handled follow-up 1",
			"handled follow-up 2",
		]);
	});

	it("delivers all steering messages in one batch in all mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		harness.session.setSteeringMode("all");
		let batchedUserMessages: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				batchedUserMessages = context.messages
					.filter((message) => message.role === "user")
					.map((message) => getMessageText(message));
				return fauxAssistantMessage("batched steer response");
			},
		]);

		await waitForToolStart;
		await harness.session.steer("steer 1");
		await harness.session.steer("steer 2");
		releaseToolExecution();
		await promptPromise;

		expect(batchedUserMessages).toEqual(["start", "steer 1", "steer 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "batched steer response"]);
	});

	it("delivers all follow-up messages in one batch in all mode", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		harness.session.setFollowUpMode("all");
		let batchedUserMessages: string[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			(context) => {
				batchedUserMessages = context.messages
					.filter((message) => message.role === "user")
					.map((message) => getMessageText(message));
				return fauxAssistantMessage("batched follow-up response");
			},
		]);

		await waitForToolStart;
		await harness.session.followUp("follow-up 1");
		await harness.session.followUp("follow-up 2");
		releaseToolExecution();
		await promptPromise;

		expect(batchedUserMessages).toEqual(["start", "follow-up 1", "follow-up 2"]);
		expect(getAssistantTexts(harness)).toEqual(["", "original turn complete", "batched follow-up response"]);
	});

	it("queues custom messages with deliverAs steer while streaming", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		let sawCustomMessage = false;

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "steer custom"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await waitForToolStart;
		await harness.session.sendCustomMessage(
			{ customType: "queue-test", content: "steer custom", display: true, details: { value: 1 } },
			{ deliverAs: "steer" },
		);
		releaseToolExecution();
		await promptPromise;

		expect(sawCustomMessage).toBe(true);
		expect(
			harness.session.messages.some((message) => message.role === "custom" && message.customType === "queue-test"),
		).toBe(true);
	});

	it("queues custom messages with deliverAs followUp while streaming", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		let sawCustomMessage = false;

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("original turn complete"),
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "follow-up custom"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await waitForToolStart;
		await harness.session.sendCustomMessage(
			{ customType: "queue-test", content: "follow-up custom", display: true, details: { value: 1 } },
			{ deliverAs: "followUp" },
		);
		releaseToolExecution();
		await promptPromise;

		expect(sawCustomMessage).toBe(true);
		expect(
			harness.session.messages.some((message) => message.role === "custom" && message.customType === "queue-test"),
		).toBe(true);
	});

	it("injects nextTurn custom messages into the next prompt", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		let sawCustomMessage = false;

		await harness.session.sendCustomMessage(
			{ customType: "next-turn", content: "carry this", display: true, details: {} },
			{ deliverAs: "nextTurn" },
		);

		harness.setResponses([
			(context) => {
				sawCustomMessage = context.messages.some(
					(message) =>
						message.role === "user" &&
						typeof message.content !== "string" &&
						message.content.some((part) => part.type === "text" && part.text === "carry this"),
				);
				return fauxAssistantMessage("done");
			},
		]);

		await harness.session.prompt("normal prompt");

		expect(sawCustomMessage).toBe(true);
		expect(harness.session.messages.map((message) => message.role)).toEqual([
			"system",
			"user",
			"custom",
			"assistant",
		]);
	});

	it("updates pendingMessageCount and removes queued text before message_start is emitted", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const countsAtQueuedMessageStart: number[] = [];

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		harness.session.subscribe((event) => {
			if (
				event.type === "message_start" &&
				event.message.role === "user" &&
				getMessageText(event.message) === "queued"
			) {
				countsAtQueuedMessageStart.push(harness.session.pendingMessageCount);
			}
		});

		await waitForToolStart;
		await harness.session.steer("queued");
		expect(harness.session.pendingMessageCount).toBe(1);
		releaseToolExecution();
		await promptPromise;

		expect(countsAtQueuedMessageStart).toEqual([0]);
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	it("throws when queueing an extension command with steer", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async () => {},
					});
				},
			],
		});
		harnesses.push(harness);

		await expect(harness.session.steer("/testcmd queued")).rejects.toThrow(
			'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.',
		);
	});

	it("throws when queueing an extension command with followUp", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.registerCommand("testcmd", {
						description: "Test command",
						handler: async () => {},
					});
				},
			],
		});
		harnesses.push(harness);

		await expect(harness.session.followUp("/testcmd queued")).rejects.toThrow(
			'Extension command "/testcmd" cannot be queued. Use prompt() or execute the command when not streaming.',
		);
	});

	// A queued image used to reach the provider untouched: prompt() returns into the queue before the
	// idle path normalizes, and steer()/followUp() never normalized at all, so auto-resize was skipped
	// for every image pasted into a running turn.
	it.each(["steer", "followUp"] as const)("normalizes images queued with %s() while streaming", async (behavior) => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		processImage.mockResolvedValueOnce({
			ok: true,
			data: Buffer.from("normalized").toString("base64"),
			mimeType: "image/png",
			hints: [DIMENSION_HINT],
		});

		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		await (behavior === "steer"
			? harness.session.steer("queued with image", [OVERSIZED_IMAGE])
			: harness.session.followUp("queued with image", [OVERSIZED_IMAGE]));
		releaseToolExecution();
		await promptPromise;

		const queued = harness.session.messages.find(
			(message): message is Extract<AgentMessage, { role: "user" }> =>
				message.role === "user" && getMessageText(message).startsWith("queued with image"),
		);
		expect(processImage).toHaveBeenCalledWith(
			expect.any(Uint8Array),
			"image/png",
			expect.objectContaining({ autoResizeImages: true }),
		);
		expect(queued?.content).toContainEqual({
			type: "image",
			data: Buffer.from("normalized").toString("base64"),
			mimeType: "image/png",
		});
		expect(getMessageText(queued)).toBe(`queued with image\n\n${DIMENSION_HINT}`);
	});

	// The three ways a submission can reach the queue while a run is active.
	const SUBMISSIONS: Array<[string, (harness: Harness) => Promise<unknown>]> = [
		["steer()", (harness) => harness.session.steer("queued with image", [OVERSIZED_IMAGE])],
		["followUp()", (harness) => harness.session.followUp("queued with image", [OVERSIZED_IMAGE])],
		[
			"prompt({streamingBehavior})",
			(harness) =>
				harness.session.prompt("queued with image", {
					images: [OVERSIZED_IMAGE],
					streamingBehavior: "steer",
				}),
		],
	];

	/** Hold the next `processImage` call open, and return the function that lets it finish. */
	function deferResize(): () => void {
		let release: () => void = () => {};
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		processImage.mockImplementationOnce(async (_bytes, mimeType) => {
			await gate;
			return {
				ok: true,
				data: Buffer.from("normalized").toString("base64"),
				mimeType,
				hints: [DIMENSION_HINT],
			};
		});
		return () => release();
	}

	// A submission whose image is still being resized belongs to the run that is active when it is
	// made: the queue is drained by that run, so settling before the message is registered would
	// strand it until some later run - which may never come.
	it.each(SUBMISSIONS)("delivers an image submitted with %s inside the running turn", async (_name, submit) => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		const submitted = submit(harness);
		// The resize outlives the tool call that was holding the turn open, so the run reaches its
		// settle decision while the submission is still being prepared.
		releaseToolExecution();
		await new Promise((resolve) => setTimeout(resolve, 25));
		expect(harness.session.isStreaming).toBe(true);
		releaseResize();
		await submitted;
		await promptPromise;

		// Delivered with the hint, drained, and by the run that was already active rather than a
		// second one started after the fact.
		expect(getUserTexts(harness)).toContain(`queued with image\n\n${DIMENSION_HINT}`);
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	it.each(SUBMISSIONS)(
		"drops an image submitted with %s when the run is aborted while it resizes",
		async (_name, submit) => {
			const waiting = await createWaitingHarness();
			const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
			harnesses.push(harness);
			const releaseResize = deferResize();
			harness.setResponses([
				fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
				fauxAssistantMessage("should not run"),
			]);

			await waitForToolStart;
			const submitted = submit(harness);
			const aborting = harness.session.abort();
			releaseToolExecution();
			releaseResize();
			await submitted;
			await aborting;
			await promptPromise;

			expect(getUserTexts(harness).some((text) => text.includes("queued with image"))).toBe(false);
			expect(harness.session.pendingMessageCount).toBe(0);
			expect(getAssistantTexts(harness)).not.toContain("should not run");
		},
	);

	// Escape must not wait for the image: abort() waits for this session to go idle, so a settlement
	// wait that blocks on a resize would make the abort hang for as long as the resize takes.
	it("does not make abort wait for an in-flight resize", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		const submitted = harness.session.steer("queued with image", [OVERSIZED_IMAGE]);
		releaseToolExecution();
		await new Promise((resolve) => setTimeout(resolve, 25));

		const aborting = harness.session.abort();
		const outcome = await Promise.race([
			aborting.then(() => "aborted"),
			new Promise((resolve) => setTimeout(() => resolve("still waiting"), 50)),
		]);
		expect(outcome).toBe("aborted");

		releaseResize();
		await Promise.all([submitted, aborting, promptPromise]);
	});

	// Clearing the queues cancels in-flight work, so settlement must not keep waiting on it.
	it("lets a cleared submission stop holding settlement", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		const submitted = harness.session.steer("queued with image", [OVERSIZED_IMAGE]);
		releaseToolExecution();
		await new Promise((resolve) => setTimeout(resolve, 25));
		harness.session.clearQueue();

		const outcome = await Promise.race([
			promptPromise.then(() => "settled"),
			// Generous: with the defect present the run keeps waiting for the resize released below.
			new Promise((resolve) => setTimeout(() => resolve("held"), 2_000)),
		]);
		expect(outcome).toBe("settled");

		releaseResize();
		await submitted;
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	// An aborted submission must not make the next run wait for its resize either.
	it("does not make a later run wait on an aborted submission", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("first"),
			fauxAssistantMessage("second"),
			fauxAssistantMessage("third"),
			fauxAssistantMessage("fourth"),
		]);

		await waitForToolStart;
		const submitted = harness.session.steer("queued with image", [OVERSIZED_IMAGE]);
		const aborting = harness.session.abort();
		releaseToolExecution();
		await aborting;
		await promptPromise.catch(() => {});

		const second = harness.session.prompt("second");
		const outcome = await Promise.race([
			second.then(() => "done"),
			// Generous: with the defect present this run waits for the resize, which is never released here.
			new Promise((resolve) => setTimeout(() => resolve("waiting"), 2_000)),
		]);
		expect(outcome).toBe("done");

		releaseResize();
		await submitted;
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	// The last settlement decision has its own async gap: an extension handler can hold the boundary
	// open, and a submission made during it belongs to this run just as much as the earlier ones.
	it("drains a submission made while the settlement boundary is held open", async () => {
		const boundaryStarted = deferred();
		const releaseBoundary = deferred();
		const waiting = await createWaitingHarness({
			extensionFactories: [
				(pi) => {
					pi.on("agent_before_settle", async () => {
						boundaryStarted.resolve();
						await releaseBoundary.promise;
					});
				},
			],
		});
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		releaseToolExecution();
		await boundaryStarted.promise;
		const submitted = harness.session.steer("queued with image", [OVERSIZED_IMAGE]);
		releaseBoundary.resolve();
		releaseResize();
		await submitted;
		await promptPromise;

		expect(getUserTexts(harness).some((text) => text.includes("queued with image"))).toBe(true);
		expect(harness.session.pendingMessageCount).toBe(0);
	});

	// RPC answers a prompt request only through preflightResult or a rejection, so a cancelled
	// submission has to report a disposition or the request is never answered.
	it("reports a submission cancelled during resizing as handled", async () => {
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		const releaseResize = deferResize();
		const dispositions: string[] = [];
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		const submitted = harness.session.prompt("queued with image", {
			images: [OVERSIZED_IMAGE],
			streamingBehavior: "steer",
			preflightResult: (disposition) => dispositions.push(disposition),
		});
		const aborting = harness.session.abort();
		releaseToolExecution();
		releaseResize();
		await submitted;
		await aborting;
		await promptPromise;

		expect(dispositions).toEqual(["handled"]);
	});

	it("omits an image that cannot be resized and tells the model why", async () => {
		const OMITTED = "[Image omitted: could not be resized below the inline image size limit.]";
		const waiting = await createWaitingHarness();
		const { harness, waitForToolStart, promptPromise, releaseToolExecution } = waiting;
		harnesses.push(harness);
		processImage.mockResolvedValueOnce({ ok: false, message: OMITTED });
		harness.setResponses([
			fauxAssistantMessage(fauxToolCall("wait", {}), { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);

		await waitForToolStart;
		await harness.session.steer("queued with image", [OVERSIZED_IMAGE]);
		releaseToolExecution();
		await promptPromise;

		const queued = harness.session.messages.find(
			(message): message is Extract<AgentMessage, { role: "user" }> =>
				message.role === "user" && getMessageText(message).includes("queued with image"),
		);
		expect(queued?.content).not.toContainEqual(expect.objectContaining({ type: "image" }));
		expect(getMessageText(queued)).toBe(`queued with image\n\n${OMITTED}`);
	});

	it("delivers follow-ups queued during agent_end", async () => {
		let sent = false;
		const harness = await createHarness({
			extensionFactories: [
				(pi: ExtensionAPI) => {
					pi.on("agent_end", async () => {
						if (sent) return;
						sent = true;
						pi.sendUserMessage("conflict report", { deliverAs: "followUp" });
					});
				},
			],
		});
		harnesses.push(harness);

		harness.setResponses([fauxAssistantMessage("reply"), fauxAssistantMessage("follow-up reply")]);

		await harness.session.prompt("hello");
		await harness.session.agent.waitForIdle();

		expect(getUserTexts(harness)).toEqual(["hello", "conflict report"]);
	});
});

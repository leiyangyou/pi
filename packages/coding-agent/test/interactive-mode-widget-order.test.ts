import { beforeAll, describe, expect, test, vi } from "vitest";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

/**
 * A widget key keeps the position it first took in its container. Extensions that refresh a widget on
 * a timer re-set the same key; if a re-set moved the key to the end of the map, the status area would
 * reorder itself between renders (two timer-driven writers traded places every 1-2.5 s in the wild).
 */
function makeMode() {
	const extensionWidgetsAbove = new Map<string, any>();
	const extensionWidgetsBelow = new Map<string, any>();
	const renderWidgets = vi.fn();
	const fakeThis: any = { extensionWidgetsAbove, extensionWidgetsBelow, renderWidgets, ui: {} };
	return { fakeThis, extensionWidgetsAbove, extensionWidgetsBelow, renderWidgets };
}

function setWidget(fakeThis: any, key: string, content: any, options?: any): void {
	(InteractiveMode as any).prototype.setExtensionWidget.call(fakeThis, key, content, options);
}

function disposable(label: string) {
	const dispose = vi.fn();
	return { label, dispose, invalidate: vi.fn(), render: () => [label] };
}

const keys = (map: Map<string, any>) => [...map.keys()];

describe("InteractiveMode.setExtensionWidget keeps a widget's slot", () => {
	beforeAll(() => initTheme("dark"));

	test("re-setting one key does not move it to the bottom", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();

		setWidget(fakeThis, "herdr-fleet", ["fleet"]);
		setWidget(fakeThis, "background-jobs", ["jobs"]);
		setWidget(fakeThis, "herdr-fleet", ["fleet 2.5s later"]);

		expect(keys(extensionWidgetsAbove)).toEqual(["herdr-fleet", "background-jobs"]);
	});

	test("three timer-driven writers keep their order across refreshes", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();

		setWidget(fakeThis, "a", ["a"]);
		setWidget(fakeThis, "b", ["b"]);
		setWidget(fakeThis, "c", ["c"]);
		setWidget(fakeThis, "a", ["a 2"]);
		setWidget(fakeThis, "c", ["c 2"]);
		setWidget(fakeThis, "b", ["b 2"]);

		expect(keys(extensionWidgetsAbove)).toEqual(["a", "b", "c"]);
	});

	test("a widget registered late takes the bottom slot once and keeps it", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();

		setWidget(fakeThis, "first", ["first"]);
		setWidget(fakeThis, "second", ["second"]);

		setWidget(fakeThis, "first", ["first again"]);
		expect(keys(extensionWidgetsAbove)).toEqual(["first", "second"]);
	});

	test("re-setting disposes the replaced component exactly once", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();
		const first = disposable("first");

		setWidget(fakeThis, "w", () => first as any);
		setWidget(fakeThis, "w", () => disposable("second") as any);

		expect(first.dispose).toHaveBeenCalledTimes(1);
		expect(keys(extensionWidgetsAbove)).toEqual(["w"]);
	});

	test("a placement change moves the widget to the other container", () => {
		const { fakeThis, extensionWidgetsAbove, extensionWidgetsBelow } = makeMode();
		const above = disposable("above");

		setWidget(fakeThis, "a", () => above as any);
		setWidget(fakeThis, "b", ["b"]);
		setWidget(fakeThis, "a", ["a"], { placement: "belowEditor" });

		expect(keys(extensionWidgetsAbove)).toEqual(["b"]);
		expect(keys(extensionWidgetsBelow)).toEqual(["a"]);
		expect(above.dispose).toHaveBeenCalledTimes(1);
	});

	test("clearing removes the key from both containers and disposes it", () => {
		const { fakeThis, extensionWidgetsAbove, extensionWidgetsBelow } = makeMode();
		const component = disposable("w");

		setWidget(fakeThis, "w", () => component as any);
		setWidget(fakeThis, "w", undefined);

		expect(keys(extensionWidgetsAbove)).toEqual([]);
		expect(keys(extensionWidgetsBelow)).toEqual([]);
		expect(component.dispose).toHaveBeenCalledTimes(1);
	});

	test("a factory that throws leaves the previous widget in place", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();
		const first = disposable("first");

		setWidget(fakeThis, "w", () => first as any);
		expect(() =>
			setWidget(fakeThis, "w", () => {
				throw new Error("factory failed");
			}),
		).toThrow("factory failed");

		// nothing was disposed or removed, so the last good widget keeps rendering
		expect(keys(extensionWidgetsAbove)).toEqual(["w"]);
		expect(first.dispose).not.toHaveBeenCalled();
	});

	test("a factory that clears its own key while running disposes the old widget once", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();
		const first = disposable("first");

		setWidget(fakeThis, "w", () => first as any);
		setWidget(fakeThis, "w", () => {
			setWidget(fakeThis, "w", undefined); // re-entrant clear from inside the factory
			return disposable("second") as any;
		});

		expect(first.dispose).toHaveBeenCalledTimes(1);
		expect(keys(extensionWidgetsAbove)).toEqual(["w"]);

		// and the component must not be disposed a second time by a later clear
		setWidget(fakeThis, "w", undefined);
		expect(first.dispose).toHaveBeenCalledTimes(1);
	});

	test("an old component whose dispose throws is still replaced, and the error surfaces", () => {
		const { fakeThis, extensionWidgetsAbove, renderWidgets } = makeMode();
		const first = disposable("first");
		const disposeError = new Error("dispose failed");
		first.dispose = vi.fn(() => {
			throw disposeError;
		});
		const second = disposable("second");

		setWidget(fakeThis, "w", () => first as any);
		renderWidgets.mockClear();

		let caught: unknown;
		try {
			setWidget(fakeThis, "w", () => second as any);
		} catch (error) {
			caught = error;
		}

		// identity, not merely "something threw": a comma-expression slip would surface undefined here
		expect(caught).toBe(disposeError);
		expect(extensionWidgetsAbove.get("w")).toBe(second);
		expect(first.dispose).toHaveBeenCalledTimes(1);
		expect(renderWidgets).toHaveBeenCalledTimes(1);
	});

	test("a dispose that throws undefined is still reported", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();
		const first = disposable("first");
		first.dispose = vi.fn(() => {
			throw undefined;
		});
		const second = disposable("second");

		setWidget(fakeThis, "w", () => first as any);
		let thrown = false;
		try {
			setWidget(fakeThis, "w", () => second as any);
		} catch {
			thrown = true;
		}

		expect(thrown).toBe(true);
		expect(extensionWidgetsAbove.get("w")).toBe(second);
	});

	test("a throwing dispose does not stop a clear", () => {
		const { fakeThis, extensionWidgetsAbove, renderWidgets } = makeMode();
		const first = disposable("first");
		const disposeError = new Error("dispose failed");
		first.dispose = vi.fn(() => {
			throw disposeError;
		});

		setWidget(fakeThis, "w", () => first as any);
		renderWidgets.mockClear();

		let caught: unknown;
		try {
			setWidget(fakeThis, "w", undefined);
		} catch (error) {
			caught = error;
		}

		expect(caught).toBe(disposeError);
		expect(keys(extensionWidgetsAbove)).toEqual([]);
		expect(renderWidgets).toHaveBeenCalledTimes(1);
	});

	test("a factory that registers the instance it returns does not get it disposed", () => {
		const { fakeThis, extensionWidgetsAbove } = makeMode();
		const replacement = disposable("replacement");

		setWidget(fakeThis, "w", () => {
			setWidget(fakeThis, "w", () => replacement as any); // re-entrant: installs `replacement`
			return replacement as any; // and hands that same instance back
		});

		expect(replacement.dispose).not.toHaveBeenCalled();
		expect(extensionWidgetsAbove.get("w")).toBe(replacement);
	});

	test("each update still renders", () => {
		const { fakeThis, renderWidgets } = makeMode();

		setWidget(fakeThis, "w", ["one"]);
		setWidget(fakeThis, "w", ["two"]);
		setWidget(fakeThis, "w", undefined);

		expect(renderWidgets).toHaveBeenCalledTimes(3);
	});
});

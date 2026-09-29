import {
	cpSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DefaultPackageManager } from "../src/core/package-manager.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { ConfigSelectorComponent } from "../src/modes/interactive/components/config-selector.ts";

describe("user package extension implementation root", () => {
	let cwd: string;
	let agentDir: string;
	let original: string;
	let candidate: string;
	let marker: string;
	beforeEach(() => {
		// Canonicalise the temp root: on macOS `tmpdir()` is `/var/folders/...`, whose real path is
		// `/private/var/...`, and the loader returns canonical `resolvedPath` values. Without this the
		// expectations below compare two spellings of the same directory.
		cwd = realpathSync(mkdtempSync(join(tmpdir(), "pi-implementation-")));
		agentDir = join(cwd, "agent");
		original = join(cwd, "original");
		candidate = join(cwd, "candidate");
		marker = join(cwd, "evaluated");
		mkdirSync(agentDir);
		mkdirSync(join(cwd, ".pi"));
		for (const [root, label] of [
			[original, "original"],
			[candidate, "candidate"],
		]) {
			mkdirSync(join(root, "skills", "sample"), { recursive: true });
			mkdirSync(join(root, "prompts"));
			writeFileSync(
				join(root, "package.json"),
				JSON.stringify({
					name: "fixture-policy",
					version: "1.0.0",
					type: "module",
					pi: { extensions: ["a.js", "b.js"], skills: ["skills"], prompts: ["prompts"] },
				}),
			);
			writeFileSync(join(root, "helper.js"), `export default ${JSON.stringify(label)};`);
			writeFileSync(join(root, "asset.txt"), label);
			writeFileSync(
				join(root, "skills/sample/SKILL.md"),
				"---\nname: sample\ndescription: synthetic\n---\nSynthetic skill",
			);
			writeFileSync(join(root, "prompts/sample.md"), "Synthetic prompt");
			for (const entry of ["a", "b"])
				writeFileSync(
					join(root, `${entry}.js`),
					`import {appendFileSync,readFileSync} from 'node:fs'; import value from './helper.js'; appendFileSync(${JSON.stringify(marker)},value+':${entry}:module:'+readFileSync(new URL('./asset.txt',import.meta.url),'utf8')+'\\n'); export default pi => { appendFileSync(${JSON.stringify(marker)},value+':${entry}:factory\\n'); pi.on('before_agent_start',()=>{}); };`,
				);
		}
	});
	afterEach(() => {
		vi.restoreAllMocks();
		rmSync(cwd, { recursive: true, force: true });
	});
	function settings(projectPackages?: unknown[]) {
		writeFileSync(
			join(agentDir, "settings.json"),
			JSON.stringify({ packages: [{ source: original, extensionImplementationRoot: candidate }] }),
		);
		if (projectPackages) writeFileSync(join(cwd, ".pi/settings.json"), JSON.stringify({ packages: projectPackages }));
		return SettingsManager.create(cwd, agentDir, { projectTrusted: true });
	}
	function loader(manager = settings()) {
		return new DefaultResourceLoader({
			cwd,
			agentDir,
			settingsManager: manager,
			noThemes: true,
			noPromptTemplates: true,
		});
	}
	function evaluated() {
		try {
			return readFileSync(marker, "utf8");
		} catch {
			return "";
		}
	}
	it("loads candidate relative imports and reads with original attribution and unchanged resources", async () => {
		const manager = settings();
		const resolver = new DefaultPackageManager({ cwd, agentDir, settingsManager: manager });
		const mapped = await resolver.resolve();
		manager.setPackages([original]);
		const baseline = await resolver.resolve();
		for (const key of ["skills", "prompts", "themes"] as const)
			expect(mapped[key].map((r) => [r.path, r.enabled])).toEqual(baseline[key].map((r) => [r.path, r.enabled]));
		const resources = loader(settings());
		await resources.reload();
		expect(evaluated()).toContain("candidate:a:module:candidate");
		expect(evaluated()).not.toContain("original:");
		const extension = resources.getExtensions().extensions[0];
		expect(extension.path).toBe(join(original, "a.js"));
		expect(extension.resolvedPath).toBe(join(candidate, "a.js"));
		expect(extension.sourceInfo?.baseDir).toBe(candidate);
		expect(extension.sourceInfo?.source).toBe(original);
	});
	it("preserves selector through config toggles, settings writes, and pin updates", async () => {
		const manager = settings();
		const resolver = new DefaultPackageManager({ cwd, agentDir, settingsManager: manager });
		const resolved = await resolver.resolve();
		const selector = new ConfigSelectorComponent(
			{ global: resolved, project: resolved },
			manager,
			cwd,
			agentDir,
			() => {},
			() => {},
			() => {},
			24,
			"global",
		);
		selector.getResourceList().handleInput(" ");
		selector.getResourceList().handleInput(" ");
		expect(manager.getGlobalSettings().packages).toEqual([
			{ source: original, extensionImplementationRoot: candidate, extensions: ["+a.js"] },
		]);
		manager.setTheme("light");
		await manager.reload();
		expect(manager.getGlobalSettings().packages).toEqual([
			{ source: original, extensionImplementationRoot: candidate, extensions: ["+a.js"] },
		]);
		manager.setPackages([{ source: "git:github.com/pi-fixture/policy@v1", extensionImplementationRoot: candidate }]);
		resolver.addSourceToSettings("git:github.com/pi-fixture/policy@v2");
		await manager.reload();
		expect(manager.getGlobalSettings().packages).toEqual([
			{ source: "git:github.com/pi-fixture/policy@v2", extensionImplementationRoot: candidate },
		]);
	});
	it("preserves full project exclusion with expected zero executable entries", async () => {
		const resources = loader(settings([{ source: original, extensions: [] }]));
		await resources.reload();
		expect(resources.getExtensions().extensions).toHaveLength(0);
		expect(evaluated()).toBe("");
	});
	it("full project replacement loads original and ignores project selector visibly", async () => {
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = loader(settings([{ source: original, extensionImplementationRoot: candidate }]));
		await resources.reload();
		expect(evaluated()).toContain("original:a:factory");
		expect(evaluated()).not.toContain("candidate:");
		expect(diagnostic).toHaveBeenCalled();
	});
	it.each([
		["-a.js", "b.js"],
		["+a.js", "a.js"],
	])("applies autoload:false delta %s over user selector", async (pattern, expected) => {
		const manager = settings([{ source: original, autoload: false, extensions: [pattern] }]);
		if (pattern.startsWith("+"))
			manager.setPackages([{ source: original, extensionImplementationRoot: candidate, extensions: [] }]);
		const resources = loader(manager);
		await resources.reload();
		expect(resources.getExtensions().extensions.map((e) => e.resolvedPath)).toEqual([join(candidate, expected)]);
	});
	it.each([{ filters: [] }, { filters: ["+a.js", "-b.js"] }])(
		"honors user filters $filters before mapping",
		async ({ filters }) => {
			const manager = settings();
			manager.setPackages([{ source: original, extensionImplementationRoot: candidate, extensions: filters }]);
			const resources = loader(manager);
			await resources.reload();
			expect(resources.getExtensions().extensions).toHaveLength(filters.length ? 1 : 0);
			expect(evaluated()).not.toContain("candidate:b:");
		},
	);
	it.each(["missing", "escape", "version", "duplicate", "relative", "shared", "ancestor"])(
		"falls back whole package before candidate evaluation for %s",
		async (failure) => {
			if (failure === "missing") rmSync(join(candidate, "b.js"));
			if (failure === "escape" || failure === "duplicate") {
				rmSync(join(candidate, "b.js"));
				symlinkSync(
					failure === "escape" ? join(original, "b.js") : join(candidate, "a.js"),
					join(candidate, "b.js"),
				);
			}
			if (failure === "version")
				writeFileSync(
					join(candidate, "package.json"),
					JSON.stringify({ name: "fixture-policy", version: "2.0.0" }),
				);
			if (failure === "ancestor") {
				for (const entry of ["package.json", "a.js", "b.js", "helper.js", "asset.txt"])
					cpSync(join(candidate, entry), join(cwd, entry));
				candidate = cwd;
			}
			const manager = settings();
			if (failure === "relative")
				manager.setPackages([{ source: original, extensionImplementationRoot: "candidate" }]);
			if (failure === "shared") manager.setPackages([{ source: original, extensionImplementationRoot: original }]);
			const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
			const resources = loader(manager);
			await resources.reload();
			expect(evaluated()).not.toContain("candidate:");
			expect(evaluated()).toContain("original:a:factory");
			expect(evaluated()).toContain("original:b:factory");
			expect(diagnostic).toHaveBeenCalled();
		},
	);
	it.each(["npm", "git"])("does not inherit selector across same-identity project %s replacement", async (kind) => {
		const userSource = kind === "npm" ? "npm:fixture-policy@1.0.0" : "git:github.com/pi-fixture/policy@old";
		const projectSource = kind === "npm" ? "npm:fixture-policy@2.0.0" : "git:github.com/pi-fixture/policy@new";
		const suffix = kind === "npm" ? "npm/node_modules/fixture-policy" : "git/github.com/pi-fixture/policy";
		cpSync(original, join(agentDir, suffix), { recursive: true });
		const projectRoot = join(cwd, ".pi", suffix);
		cpSync(original, projectRoot, { recursive: true });
		if (kind === "npm") {
			const manifest = JSON.parse(readFileSync(join(projectRoot, "package.json"), "utf8"));
			manifest.version = "2.0.0";
			writeFileSync(join(projectRoot, "package.json"), JSON.stringify(manifest));
		}
		const manager = settings([projectSource]);
		manager.setPackages([{ source: userSource, extensionImplementationRoot: candidate }]);
		const resources = loader(manager);
		await resources.reload();
		expect(resources.getExtensions().extensions.map((e) => e.resolvedPath)).toEqual([
			join(projectRoot, "a.js"),
			join(projectRoot, "b.js"),
		]);
		expect(evaluated()).not.toContain("candidate:");
	});
	it("rejects cross-package executable collisions before either candidate loads", async () => {
		const other = join(cwd, "other");
		cpSync(original, other, { recursive: true });
		const manager = settings();
		manager.setPackages([original, other].map((source) => ({ source, extensionImplementationRoot: candidate })));
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = loader(manager);
		await resources.reload();
		expect(resources.getExtensions().extensions).toHaveLength(4);
		expect(evaluated()).not.toContain("candidate:");
		expect(diagnostic).toHaveBeenCalled();
	});
	it("does not collapse mapped code onto an enabled unmapped package entry", async () => {
		const manager = settings();
		manager.setPackages([{ source: original, extensionImplementationRoot: candidate }, candidate]);
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = loader(manager);
		await resources.reload();
		expect(evaluated().match(/candidate:a:factory/g)).toHaveLength(1);
		expect(evaluated()).toContain("original:a:factory");
		expect(diagnostic).toHaveBeenCalled();
	});
	it("validates executable collisions after CLI entries join the load set", async () => {
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = new DefaultResourceLoader({
			cwd,
			agentDir,
			settingsManager: settings(),
			additionalExtensionPaths: [join(candidate, "a.js")],
		});
		await resources.reload();
		expect(evaluated().match(/candidate:a:factory/g)).toHaveLength(1);
		expect(evaluated()).toContain("original:a:factory");
		expect(evaluated()).not.toContain("candidate:b:factory");
		expect(diagnostic).toHaveBeenCalled();
	});
	it("revalidates collisions introduced by whole-package fallback", async () => {
		const roots = [original, ...[1, 2, 3].map((index) => join(cwd, `cascade-${index}`))];
		for (const root of roots.slice(1)) cpSync(original, root, { recursive: true });
		const manager = settings();
		manager.setPackages(
			roots.map((source, index) => ({
				source,
				...(index < 3 ? { extensionImplementationRoot: roots[index + 1] } : {}),
			})),
		);
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = loader(manager);
		await resources.reload();
		for (const extension of resources.getExtensions().extensions) expect(extension.resolvedPath).toBe(extension.path);
		expect(resources.getExtensions().extensions).toHaveLength(8);
		expect(diagnostic).toHaveBeenCalledTimes(1);
	});
	it.each(["npm", "git"])("rejects root inside shared %s tree", async (kind) => {
		const shared = join(agentDir, kind, "candidate");
		cpSync(candidate, shared, { recursive: true });
		const manager = settings();
		manager.setPackages([{ source: original, extensionImplementationRoot: shared }]);
		const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
		const resources = loader(manager);
		await resources.reload();
		expect(evaluated()).not.toContain("candidate:");
		expect(diagnostic).toHaveBeenCalled();
	});
	it("reuses candidate pre-trust imports without repeating factories", async () => {
		const manager = settings();
		manager.setProjectTrusted(false);
		const resources = loader(manager);
		await resources.reload({ resolveProjectTrust: async () => true });
		expect(evaluated().match(/candidate:a:factory/g)).toHaveLength(1);
	});
	it("retains existing pre-trust import parity even when project later excludes package", async () => {
		const manager = settings([{ source: original, extensions: [] }]);
		manager.setProjectTrusted(false);
		const resources = loader(manager);
		await resources.reload({ resolveProjectTrust: async () => true });
		expect(evaluated()).toContain("candidate:a:factory");
		expect(resources.getExtensions().extensions).toHaveLength(0);
	});
	it.each(["import", "factory", "export"])(
		"rejects SDK initialization rather than omitting mapped policy on %s failure",
		async (failure) => {
			writeFileSync(
				join(candidate, "a.js"),
				failure === "import"
					? "throw new Error('SECRET_FAILURE'); export default ()=>{};"
					: failure === "factory"
						? "export default ()=>{throw new Error('SECRET_FAILURE')};"
						: "export const noFactory=true;",
			);
			const manager = settings();
			const resources = loader(manager);
			await expect(resources.reload()).rejects.toThrow("Extension implementation");
			expect(() => resources.getExtensions()).toThrow("Extension implementation");
			await expect(
				createAgentSession({
					cwd,
					agentDir,
					settingsManager: manager,
					resourceLoader: resources,
					sessionManager: SessionManager.inMemory(cwd),
				}),
			).rejects.toThrow("Extension implementation");
			expect(evaluated()).not.toContain("original:");
		},
	);
});

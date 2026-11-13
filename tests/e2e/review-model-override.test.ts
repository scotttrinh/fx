import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FX_BIN, runFx } from "../evals/eval-helpers";
import {
  fakeGatewayDecision,
  fakeGatewayFinalText,
  fakeShellRun,
  startFakeGateway,
  type FakeGatewayOptions,
} from "./tmux-helpers";

const TIMEOUT = 30_000;
const MODEL = "openai/gpt-5";
const EXECUTED_MARKER = "REVIEW_OVERRIDE_EXECUTED";
const DECISION_MODEL = "typesafe-ai/jev";
const GATEWAY_KEY = "fake-review-model-override-key";

type IsolatedRoot = {
  root: string;
  home: string;
  workspace: string;
};

const roots: string[] = [];
const gateways: Array<{ stop(): void }> = [];

afterEach(() => {
  for (const gateway of gateways.splice(0)) gateway.stop();
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

function createIsolatedRoot(settings: Record<string, unknown> = {}): IsolatedRoot {
  const root = realpathSync(
    mkdtempSync(join(tmpdir(), "fx-review-model-override-e2e-")),
  );
  const home = join(root, "home");
  const workspace = join(root, "workspace");
  mkdirSync(join(home, ".fx"), { recursive: true });
  mkdirSync(workspace, { recursive: true });
  writeFileSync(
    join(home, ".fx", "settings.json"),
    JSON.stringify({ sandbox: "none", permission: {}, ...settings }),
  );
  roots.push(root);
  return { root, home, workspace: realpathSync(workspace) };
}

// Runs one auto-mode turn whose model asks to print the marker, then ends
// with `finalText`. The review of that command is the test's subject.
async function runReviewedCommand(options: {
  settings?: Record<string, unknown>;
  env?: Record<string, string | undefined>;
  gateway?: FakeGatewayOptions;
  finalText?: string;
}) {
  const root = createIsolatedRoot(options.settings);
  const gateway = startFakeGateway(
    [
      fakeShellRun("cmd_1", `printf '%s' "${EXECUTED_MARKER}"`),
      fakeGatewayFinalText(options.finalText ?? "Ran the command."),
    ],
    options.gateway,
  );
  gateways.push(gateway);
  const result = await runFx(
    ["ask", "--auto", "--quiet", "--json", "Print the marker."],
    {
      cwd: root.workspace,
      env: {
        HOME: root.home,
        AI_GATEWAY_API_KEY: GATEWAY_KEY,
        VERCEL_OIDC_TOKEN: undefined,
        FX_GATEWAY_BASE_URL: gateway.baseUrl,
        FX_GATEWAY_CHAT_URL: gateway.chatUrl,
        FX_MODEL: MODEL,
        FX_PERMISSION_MODE: "auto",
        FX_AUTO_UPGRADE: "0",
        FX_REVIEW_MODEL: undefined,
        NO_COLOR: "1",
        ...options.env,
      },
      timeoutMs: TIMEOUT,
    },
  );
  return { result, gateway };
}

describe("review model override", () => {
  test.each([
    ["FX_REVIEW_MODEL", {}, { FX_REVIEW_MODEL: DECISION_MODEL }],
    ["review_model in settings.json", { review_model: DECISION_MODEL }, {}],
  ])(
    "%s naming a decision model sends the review to the gateway decision endpoint and runs a cleared action",
    async (_name, settings, env) => {
      const { result, gateway } = await runReviewedCommand({ settings, env });
      expect(result.code).toBe(0);
      expect(result.stdout).toContain(EXECUTED_MARKER);
      expect(gateway.classifierRequests).toHaveLength(0);
      expect(gateway.decisionRequests).toHaveLength(1);

      const { body, headers } = gateway.decisionRequests[0];
      expect(headers.get("authorization")).toBe(`Bearer ${GATEWAY_KEY}`);
      expect(headers.get("ai-model-id")).toBe(DECISION_MODEL);
      expect(headers.get("ai-decision-model-specification-version")).toBe("4");
      const review = JSON.parse(body);
      expect(review.stateParts).toHaveLength(1);
      expect(review.stateParts[0].type).toBe("json");
      const state = review.stateParts[0].value;
      expect(state.review_policy.length).toBeGreaterThan(100);
      expect(state.pending_action_tool).toBe("shell");
      expect(state.pending_action_arguments).toContain(EXECUTED_MARKER);
      expect(review.questions.decision.type).toBe("choice");
      expect(Object.keys(review.questions.decision.criteria)).toEqual([
        "clear",
        "caution",
      ]);
    },
    TIMEOUT,
  );

  test(
    "a decision model's caution holds the action and the loop continues",
    async () => {
      const { result, gateway } = await runReviewedCommand({
        env: { FX_REVIEW_MODEL: DECISION_MODEL },
        gateway: { decisionResponse: () => fakeGatewayDecision("caution") },
        finalText: "Understood, leaving it unexecuted.",
      });
      expect(result.code).toBe(0);
      expect(result.stdout).not.toContain(EXECUTED_MARKER);
      expect(result.stdout).toContain("Understood, leaving it unexecuted.");
      expect(gateway.decisionRequests).toHaveLength(1);
    },
    TIMEOUT,
  );

  test.each([
    ["fails", () => new Response("upstream unavailable", { status: 500 })],
    ["answers off the question", () => Response.json({ answers: { decision: { type: "choice", choice: "maybe" } } })],
    ["declines to answer", () => Response.json({ answers: { decision: { type: "refusal" } } })],
    ["answers with a truncated body", () => new Response('{"answers":{"decisio', { headers: { "content-type": "application/json" } })],
  ])(
    "a decision endpoint that %s holds the action instead of running it unreviewed",
    async (_name, decisionResponse) => {
      const { result, gateway } = await runReviewedCommand({
        env: { FX_REVIEW_MODEL: DECISION_MODEL },
        gateway: { decisionResponse },
        finalText: "Review unavailable, holding.",
      });
      expect(result.code, `stdout: ${result.stdout}\nstderr: ${result.stderr}`).toBe(0);
      expect(result.stdout).not.toContain(EXECUTED_MARKER);
      expect(gateway.decisionRequests.length).toBeGreaterThan(0);
      expect(gateway.classifierRequests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test.each([
    ["review_model naming a language model reviews on the chat endpoint with that model", { review_model: "openai/gpt-5-alt" }, "openai/gpt-5-alt"],
    ["no review model reviews with the default language model", {}, "openai/gpt-5.6-luna"],
  ])(
    "%s",
    async (_name, settings, reviewer) => {
      const { result, gateway } = await runReviewedCommand({ settings });
      expect(result.code).toBe(0);
      expect(result.stdout).toContain(EXECUTED_MARKER);
      expect(gateway.decisionRequests).toHaveLength(0);
      expect(gateway.classifierRequests).toHaveLength(1);
      expect(
        gateway.classifierRequests[0].headers.get("ai-language-model-id"),
      ).toBe(reviewer);
    },
    TIMEOUT,
  );
});

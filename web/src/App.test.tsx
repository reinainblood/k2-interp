// @vitest-environment jsdom
import { afterEach, beforeAll, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import App from "./App";

vi.mock("./RouterMatrix", () => ({ RouterMatrix: () => null }));
vi.mock("./api", () => ({
  api: vi.fn(() => new Promise(() => {})),
  downloadCapture: vi.fn(),
  turnPath: () => "/turn",
}));
vi.mock("./useWorkbench", () => ({
  useWorkbench: () => ({
    authorized: true,
    checkpoints: ["sft_3_1500"],
    sessions: [],
    session: {
      id: "session-test",
      checkpoint: "sft_3_1500",
      system: "",
      turns: [],
      captures: [],
      created_at: "2026-10-22",
    },
    runId: "run-test",
    frame: {
      run_id: "run-test",
      status: "complete",
      sequence: 1,
      position: 3,
      input_tokens: 2,
      token_text: ["a", "b", "c", "d"],
      input_ids: [1, 2],
      output_ids: [3, 4],
      output_logprobs: [-1, -1],
      layers: {},
    },
    error: "",
    connection: "Connected",
    sending: false,
    setError: vi.fn(),
    setRunId: vi.fn(),
    setSession: vi.fn(),
    selectSession: vi.fn(),
    create: vi.fn(),
    send: vi.fn(),
    cancel: vi.fn(),
    login: vi.fn(),
    updateSystem: vi.fn(),
  }),
}));
beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    value: vi.fn(() => ({ matches: true })),
    writable: true,
  });
});
afterEach(cleanup);

it("clears a pending token inspection when returning to Follow", async () => {
  render(<App />);
  fireEvent.change(screen.getByRole("spinbutton", { name: "Position" }), {
    target: { value: "1" },
  });
  expect(screen.getByText("Reading")).toBeTruthy();
  expect(
    (screen.getByRole("button", { name: "Capture ↓" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  fireEvent.click(screen.getByRole("button", { name: "Follow" }));
  await waitFor(() =>
    expect(
      (screen.getByRole("button", { name: "Capture ↓" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false),
  );
});

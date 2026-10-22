// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ExpertGraph } from "./DetailGraphs";
import { Settings, defaults } from "./Settings";
import { tokenLabel } from "./types";
afterEach(cleanup);

describe("literal data views", () => {
  it("distinguishes uncollected expert outputs from measured zero", () => {
    render(
      <ExpertGraph
        layer={30}
        value={{
          expert_ids: [7, 12],
          expert_weights: [1.4, 1.1],
          expert_norms: { "7": 0 },
        }}
      />,
    );
    expect(screen.getByText("E007")).toBeTruthy();
    expect(screen.getByText("E012")).toBeTruthy();
    expect(screen.getByText("0")).toBeTruthy();
    expect(screen.getByText("·")).toBeTruthy();
    expect(screen.getByText("Σw 2.5")).toBeTruthy();
  });
  it("shows dense layers without inventing expert routes", () => {
    render(<ExpertGraph layer={0} />);
    expect(screen.getByText("Dense MLP")).toBeTruthy();
    expect(screen.queryByText("E000")).toBeNull();
  });
  it("keeps tokenizer spellings reversible at the ID layer", () => {
    expect(tokenLabel("ĠhandĊ")).toBe(" hand\n");
  });
});

describe("editable runtime settings", () => {
  it("applies an edited system prompt and offers native reasoning choices", async () => {
    const apply = vi.fn().mockResolvedValue(undefined);
    const change = vi.fn();
    render(
      <Settings
        value={defaults}
        onChange={change}
        system="Initial system"
        onSystem={apply}
        disabled={false}
      />,
    );
    fireEvent.change(screen.getByLabelText("System prompt"), {
      target: { value: "Revised system" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply system" }));
    expect(apply).toHaveBeenCalledWith("Revised system");
    fireEvent.change(screen.getByLabelText("Reasoning effort"), {
      target: { value: "low" },
    });
    expect(change).toHaveBeenCalledWith({
      ...defaults,
      reasoning_effort: "low",
    });
  });
  it("locks changes during an active turn", () => {
    render(
      <Settings
        value={defaults}
        onChange={vi.fn()}
        system="System"
        onSystem={vi.fn()}
        disabled
      />,
    );
    expect(
      (screen.getByLabelText("System prompt") as HTMLTextAreaElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByLabelText("Reasoning effort") as HTMLSelectElement).disabled,
    ).toBe(true);
  });
});

export async function api<T>(
  path: string,
  data?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch("/api" + path, {
    method: data === undefined ? "GET" : "POST",
    credentials: "same-origin",
    signal,
    headers: { "Content-Type": "application/json", "X-K2-Intent": "workbench" },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
export const turnPath = (sessionId: string, runId: string) =>
  `/sessions/${sessionId}/turns/${runId}`;
export async function downloadCapture(path: string, data: unknown) {
  const response = await fetch("/api" + path + "/captures", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-K2-Intent": "workbench" },
    body: JSON.stringify(data),
    credentials: "same-origin",
  });
  if (!response.ok)
    throw new Error((await response.json()).detail || response.statusText);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download =
    response.headers
      .get("content-disposition")
      ?.match(/filename="([^"]+)"/)?.[1] || "k2-capture.zip";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

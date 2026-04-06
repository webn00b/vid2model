import { performance } from "node:perf_hooks";

function installValue(name, value) {
  if (typeof globalThis[name] !== "undefined") return;
  Object.defineProperty(globalThis, name, {
    configurable: true,
    enumerable: false,
    writable: true,
    value,
  });
}

export function installHeadlessNodeRuntime() {
  installValue("self", globalThis);
  installValue("window", globalThis);
  installValue("performance", performance);
  installValue("navigator", { userAgent: "node" });
  installValue("atob", (value) => Buffer.from(String(value), "base64").toString("binary"));
  installValue("btoa", (value) => Buffer.from(String(value), "binary").toString("base64"));
  installValue(
    "createImageBitmap",
    async () => ({
      width: 1,
      height: 1,
      close() {},
    })
  );
  installValue(
    "ImageBitmap",
    class ImageBitmapMock {
      constructor(width = 1, height = 1) {
        this.width = width;
        this.height = height;
      }

      close() {}
    }
  );
}

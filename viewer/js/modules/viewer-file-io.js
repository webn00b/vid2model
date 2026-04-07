export function createViewerFileIo({
  setStatus,
  loadBvhText,
  animationListEl,
}) {
  async function loadBvhFileByName(filename) {
    setStatus(`Loading ${filename} ...`);
    try {
      const res = await fetch(`../output/${filename}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      loadBvhText(text, `output/${filename}`);
    } catch (err) {
      console.error(err);
      setStatus(`Cannot load output/${filename}. Check file path.`);
    }
  }

  async function loadDefault() {
    setStatus("Loading output/think.bvh ...");
    try {
      const res = await fetch("../output/think.bvh");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      loadBvhText(text, "output/think.bvh");
    } catch (err) {
      console.error(err);
      setStatus("Cannot load output/think.bvh. Start local server from vid2model.");
    }
  }

  async function loadAnimationsList() {
    try {
      try {
        const listRes = await fetch("./animations-list.json");
        if (listRes.ok) {
          const data = await listRes.json();
          if (Array.isArray(data)) {
            populateAnimationsList(data.sort());
            return;
          }
        }
      } catch (e) {
        console.warn("Could not load animations-list.json:", e);
      }

      const res = await fetch("../output/");
      const html = await res.text();
      const bvhFiles = [];
      const regex = /href="([^"]*\.bvh)"/g;
      let match;
      while ((match = regex.exec(html)) !== null) {
        bvhFiles.push(match[1]);
      }

      if (bvhFiles.length > 0) {
        populateAnimationsList(bvhFiles.sort());
      }
    } catch (err) {
      console.error("Failed to load animations list:", err);
    }
  }

  function populateAnimationsList(files) {
    if (!animationListEl || files.length === 0) return;

    const currentValue = animationListEl.value;
    animationListEl.innerHTML = '<option value="">Choose animation...</option>';

    for (const filename of files) {
      const option = document.createElement("option");
      option.value = filename;
      option.textContent = filename;
      animationListEl.appendChild(option);
    }

    if (currentValue && files.includes(currentValue)) {
      animationListEl.value = currentValue;
    }
  }

  return {
    loadBvhFileByName,
    loadDefault,
    loadAnimationsList,
    populateAnimationsList,
  };
}

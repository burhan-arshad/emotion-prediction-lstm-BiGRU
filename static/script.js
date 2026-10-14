(() => {
  "use strict";

  const LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"];

  const EMOJI = {
    sadness: "😢",
    joy: "😄",
    love: "❤️",
    anger: "😠",
    fear: "😨",
    surprise: "😲"
  };

  const COLOR = {
    sadness: "#5b7fde",
    joy: "#f2b705",
    love: "#e85d75",
    anger: "#e4572e",
    fear: "#8b5fbf",
    surprise: "#17bebb"
  };

  const $ = (id) => document.getElementById(id);

  const el = {
    dot: $("statusDot"),
    status: $("serverStatusText"),
    input: $("textInput"),
    count: $("charCount"),
    btn: $("analyzeBtn"),
    clear: $("clearBtn"),
    err: $("errorMsg"),
    orb: document.querySelector(".orb"),
    orbEmoji: $("orbEmoji"),
    result: $("resultSection"),
    word: $("emotionWord"),
    emoji: $("emotionEmoji"),
    conf: $("confidenceText"),
    confFill: $("confFill"),
    echo: $("echoedText"),
    bars: $("barsContainer")
  };

  const reduced = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  let modelReady = false;
  let busy = false;
  let healthTimer;

  async function checkHealth() {
    clearTimeout(healthTimer);

    let next = 5000;

    try {
      const res = await fetch("/health", {
        headers: { Accept: "application/json" }
      });

      if (!res.ok) throw new Error();

      const data = await res.json();

      modelReady = data.model_loaded === true;

      setStatus(
        modelReady ? "live" : "warming",
        modelReady
          ? "model ready — say something"
          : "waking the model up…"
      );

      next = modelReady ? 30000 : 3000;
    } catch {
      modelReady = false;
      setStatus("down", "can't reach the server");
      next = 5000;
    }

    sync();
    healthTimer = setTimeout(checkHealth, next);
  }

  function setStatus(kind, text) {
    el.dot.className = "brand-mark " + kind;
    el.status.textContent = text;
  }

  function sync() {
    const has = el.input.value.trim().length > 0;

    el.btn.disabled = !has || !modelReady || busy;
    el.btn.title = !modelReady ? "The model isn't ready yet" : "";
    el.clear.hidden = !el.input.value;
    el.count.textContent = el.input.value.length;
  }

  function insertSample(text) {
    if (!text) return;

    el.input.value = text;
    el.input.dispatchEvent(new Event("input", { bubbles: true }));
    hideError();
    sync();
    el.input.focus();
  }

  el.input.addEventListener("input", () => {
    hideError();
    sync();
  });

  el.input.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      run();
    }
  });

  el.btn.addEventListener("click", run);

  el.clear.addEventListener("click", () => {
    el.input.value = "";
    hideError();
    sync();
    el.input.focus();
  });

  document.querySelectorAll(".chip, .sample-btn").forEach((c) => {
    c.addEventListener("click", () => {
      insertSample(c.dataset.text || c.dataset.s);
    });
  });

  async function run() {
    const text = el.input.value.trim();

    if (!text || !modelReady || busy) return;

    hideError();
    setBusy(true);

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30000);

    try {
      let res;

      try {
        res = await fetch("/predict", {
          method: "POST",
          signal: ctrl.signal,
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json"
          },
          body: JSON.stringify({ text })
        });
      } catch (e) {
        throw new Error(
          e.name === "AbortError"
            ? "That took too long. Please try again."
            : "Can't reach the server. Check your connection."
        );
      }

      if (!res.ok) {
        throw new Error(await httpMessage(res));
      }

      let data;

      try {
        data = await res.json();
      } catch {
        throw new Error("The server sent a response I couldn't read.");
      }

      const result = validate(data);

      if (!result) {
        throw new Error(
          "The server's response was in an unexpected format."
        );
      }

      render(result, text);
      setBusy(false, true);
    } catch (err) {
      setBusy(false, false);
      showError(err.message || "Something went wrong. Try again.");
    } finally {
      clearTimeout(timer);
    }
  }

  async function httpMessage(res) {
    if (res.status === 503) {
      modelReady = false;
      checkHealth();
      return "The model is still loading. Try again in a moment.";
    }

    if (res.status === 422) {
      return "That text couldn't be processed. Use 1–2,000 characters.";
    }

    if (res.status >= 500) {
      return "The server hit an error. Please try again later.";
    }

    return `Request failed (${res.status}).`;
  }

  const isP = (v) =>
    typeof v === "number" &&
    Number.isFinite(v) &&
    v >= 0 &&
    v <= 1;

  function validate(d) {
    if (
      !d ||
      !LABELS.includes(d.predicted_emotion) ||
      !isP(d.confidence)
    ) {
      return null;
    }

    const probs = d.all_probabilites;

    if (
      !probs ||
      typeof probs !== "object" ||
      Array.isArray(probs) ||
      LABELS.some((label) => !isP(probs[label]))
    ) {
      return null;
    }

    return {
      emotion: d.predicted_emotion,
      confidence: d.confidence,
      probs
    };
  }

  function mixGradient(probs) {
    const total =
      LABELS.reduce((sum, label) => sum + probs[label], 0) || 1;

    let acc = 0;

    const stops = LABELS.map((label) => {
      const start = acc * 100;
      const end = (acc += probs[label] / total) * 100;

      return `${COLOR[label]} ${start.toFixed(2)}% ${end.toFixed(2)}%`;
    });

    return `conic-gradient(from 210deg, ${stops.join(", ")})`;
  }

  function setBusy(value, success) {
    busy = value;

    el.btn.classList.toggle("loading", value);
    el.btn.querySelector(".btn-label").textContent =
      value ? "Reading…" : "Read the mood";

    el.orb.classList.toggle("thinking", value);

    if (value) {
      el.orb.style.setProperty(
        "--mix",
        mixGradient(
          Object.fromEntries(LABELS.map((label) => [label, 1]))
        )
      );

      el.orb.classList.remove("settled");
      el.orbEmoji.style.opacity = "0";
    } else if (success === false) {
      el.orb.style.removeProperty("--mix");
      el.orbEmoji.textContent = "✎";
      el.orbEmoji.style.opacity = "1";
    }

    sync();
  }

  function render(result, text) {
    const emotion = result.emotion;

    document.body.dataset.emotion = emotion;

    el.orb.style.setProperty("--mix", mixGradient(result.probs));
    el.orb.classList.add("settled");

    el.orbEmoji.textContent = EMOJI[emotion];
    el.orbEmoji.style.opacity = "1";

    el.word.textContent = emotion;
    el.emoji.textContent = EMOJI[emotion];

    el.conf.textContent =
      `${(result.confidence * 100).toFixed(1)}% model confidence`;

    el.confFill.style.width = "0%";
    el.echo.textContent = `“${text}”`;

    el.bars.replaceChildren();

    const fills = [];

    LABELS.slice()
      .sort((a, b) => result.probs[b] - result.probs[a])
      .forEach((label) => {
        const row = document.createElement("div");
        row.className =
          `bar-row bar-${label}` +
          (label === emotion ? " top" : "");

        const name = document.createElement("span");
        name.className = "bar-label";
        name.textContent = `${EMOJI[label]} ${label}`;

        const track = document.createElement("span");
        track.className = "bar-track";

        const fill = document.createElement("span");
        fill.className = "bar-fill";
        track.append(fill);

        const pct = document.createElement("span");
        pct.className = "bar-pct";
        pct.textContent =
          (result.probs[label] * 100).toFixed(1) + "%";

        row.append(name, track, pct);
        el.bars.append(row);

        fills.push([fill, result.probs[label]]);
      });

    el.result.hidden = false;
    el.result.classList.remove("entering");
    void el.result.offsetWidth;
    el.result.classList.add("entering");

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.confFill.style.width =
          result.confidence * 100 + "%";

        fills.forEach(([fill, probability], index) => {
          setTimeout(
            () => {
              fill.style.width =
                Math.max(probability * 100, 0.5) + "%";
            },
            reduced ? 0 : 60 + index * 70
          );
        });
      });
    });

    el.result.scrollIntoView({
      behavior: reduced ? "auto" : "smooth",
      block: "nearest"
    });
  }

  function showError(message) {
    el.err.textContent = message;
    el.err.hidden = false;
  }

  function hideError() {
    el.err.hidden = true;
  }

  sync();
  checkHealth();
})();
(() => {
  const preview = document.querySelector("[data-analysis-preview]");
  if (!preview) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pipelineSteps = [...preview.querySelectorAll("[data-pipeline-step]")];
  const fields = {
    coefficient: preview.querySelector("[data-coefficient]"),
    status: preview.querySelector("[data-analysis-status]"),
    primary: preview.querySelector("[data-terminal-primary]"),
    secondary: preview.querySelector("[data-terminal-secondary]"),
    tertiary: preview.querySelector("[data-terminal-tertiary]"),
    detail: preview.querySelector("[data-terminal-detail]"),
    interval: preview.querySelector("[data-interval]"),
    robustness: preview.querySelector("[data-robustness]"),
    stepCount: preview.querySelector("[data-step-count]"),
    progress: preview.querySelector("[data-progress]"),
  };

  const stages = [
    {
      name: "upload",
      status: "Ingesting",
      coefficient: "--",
      progress: 12,
      primary: "Reading wage_data.csv",
      secondary: "Validating columns and observations",
      tertiary: "Preparing numeric matrix",
      detail: "1,250 rows detected",
      interval: "--",
      robustness: "Pending",
    },
    {
      name: "configure",
      status: "Configured",
      coefficient: "+2.61",
      progress: 34,
      primary: "Mapping research variables",
      secondary: "Outcome: hourly_wage",
      tertiary: "Main predictor: education_years",
      detail: "Controls: experience, gender",
      interval: "Calculating",
      robustness: "Pending",
    },
    {
      name: "estimate",
      status: "Estimating",
      coefficient: "+2.48",
      progress: 59,
      primary: "Estimating model 3 / 3",
      secondary: "Adding control: experience",
      tertiary: "Adding control: gender",
      detail: "Coefficient remains positive",
      interval: "Calculating",
      robustness: "Testing",
    },
    {
      name: "bootstrap",
      status: "Bootstrapping",
      coefficient: "+2.48",
      progress: 84,
      primary: "Running bootstrap uncertainty",
      secondary: "4,100 / 5,000 resamples",
      tertiary: "Standard error: 0.14",
      detail: "Building 95% interval",
      interval: "[2.21, 2.76]",
      robustness: "Testing",
    },
    {
      name: "result",
      status: "Stable",
      coefficient: "+2.48",
      progress: 100,
      primary: "Analysis complete",
      secondary: "Coefficient is stable across models",
      tertiary: "95% CI excludes zero",
      detail: "Associational result — not causal proof",
      interval: "[2.21, 2.76]",
      robustness: "Stable",
    },
  ];

  let activeIndex = 0;
  let timerId = null;
  let pointerFrame = null;

  function renderStage(index) {
    const stage = stages[index];
    activeIndex = index;
    preview.dataset.stage = stage.name;
    preview.dataset.stepIndex = String(index);

    pipelineSteps.forEach((step, stepIndex) => {
      step.dataset.state = stepIndex < index
        ? "complete"
        : stepIndex === index
          ? "current"
          : "pending";
    });

    fields.coefficient.textContent = stage.coefficient;
    fields.status.textContent = stage.status;
    fields.primary.textContent = stage.primary;
    fields.secondary.textContent = stage.secondary;
    fields.tertiary.textContent = stage.tertiary;
    fields.detail.textContent = stage.detail;
    fields.interval.textContent = stage.interval;
    fields.robustness.textContent = stage.robustness;
    fields.robustness.classList.toggle("is-positive", stage.name === "result");
    fields.stepCount.textContent = `${String(index + 1).padStart(2, "0")} / 05`;
    fields.progress.style.setProperty("--progress", `${stage.progress}%`);
  }

  function stopAnimation() {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  function startAnimation() {
    stopAnimation();
    if (reducedMotion.matches || document.hidden) return;
    timerId = window.setInterval(() => {
      renderStage((activeIndex + 1) % stages.length);
    }, 1750);
  }

  function applyMotionPreference() {
    if (reducedMotion.matches) {
      stopAnimation();
      renderStage(stages.length - 1);
    } else {
      renderStage(0);
      startAnimation();
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopAnimation();
    else startAnimation();
  });

  if (typeof reducedMotion.addEventListener === "function") {
    reducedMotion.addEventListener("change", applyMotionPreference);
  } else {
    reducedMotion.addListener(applyMotionPreference);
  }

  if (!reducedMotion.matches) {
    window.addEventListener("pointermove", (event) => {
      if (pointerFrame !== null) return;
      pointerFrame = window.requestAnimationFrame(() => {
        document.documentElement.style.setProperty("--pointer-x", `${event.clientX}px`);
        document.documentElement.style.setProperty("--pointer-y", `${event.clientY}px`);
        pointerFrame = null;
      });
    }, { passive: true });
  }

  applyMotionPreference();
})();

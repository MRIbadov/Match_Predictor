const homeSelect = document.getElementById("home-team");
const awaySelect = document.getElementById("away-team");
const form = document.getElementById("predict-form");
const statusText = document.getElementById("status");
const resultPanel = document.getElementById("result-panel");
const predictButton = document.getElementById("predict-button");
const swapButton = document.getElementById("swap-button");

async function loadTeams() {
  statusText.textContent = "Loading teams...";
  const response = await fetch("/teams");
  if (!response.ok) {
    throw new Error("Could not load teams.");
  }

  const payload = await response.json();
  const teams = payload.teams || [];
  populateSelect(homeSelect, teams, "Arsenal");
  populateSelect(awaySelect, teams, "Chelsea");
  statusText.textContent = "Teams loaded. Pick a matchup to begin.";
}

function populateSelect(select, teams, preferred) {
  select.innerHTML = "";
  teams.forEach((team) => {
    const option = document.createElement("option");
    option.value = team;
    option.textContent = team;
    if (team === preferred) {
      option.selected = true;
    }
    select.appendChild(option);
  });
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function favoriteLabel(result) {
  const outcomes = [
    { label: `${result.home_team} win`, value: result.p_home_win },
    { label: "Draw", value: result.p_draw },
    { label: `${result.away_team} win`, value: result.p_away_win },
  ];
  return outcomes.sort((a, b) => b.value - a.value)[0];
}

function renderResult(result) {
  document.getElementById("match-title").textContent = `${result.home_team} vs ${result.away_team}`;
  document.getElementById("model-version").textContent = `Model version ${result.model_version}`;
  document.getElementById("home-prob").textContent = formatPercent(result.p_home_win);
  document.getElementById("draw-prob").textContent = formatPercent(result.p_draw);
  document.getElementById("away-prob").textContent = formatPercent(result.p_away_win);
  document.getElementById("home-goals").textContent = result.expected_home_goals.toFixed(2);
  document.getElementById("away-goals").textContent = result.expected_away_goals.toFixed(2);

  const favorite = favoriteLabel(result);
  document.getElementById("favorite-outcome").textContent = favorite.label;
  document.getElementById("favorite-confidence").textContent = `${formatPercent(favorite.value)} confidence`;

  const scorelineList = document.getElementById("scoreline-list");
  scorelineList.innerHTML = "";
  result.top_scorelines.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "scoreline-item";
    card.innerHTML = `<strong>${entry.scoreline}</strong><span class="scoreline-prob">${formatPercent(entry.probability)}</span>`;
    scorelineList.appendChild(card);
  });

  resultPanel.classList.remove("hidden");
}

swapButton.addEventListener("click", () => {
  const currentHome = homeSelect.value;
  homeSelect.value = awaySelect.value;
  awaySelect.value = currentHome;
  statusText.textContent = "Teams swapped.";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (homeSelect.value === awaySelect.value) {
    statusText.textContent = "Please choose two different teams.";
    return;
  }

  statusText.textContent = "Running prediction...";
  predictButton.disabled = true;
  swapButton.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        home_team: homeSelect.value,
        away_team: awaySelect.value,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    renderResult(payload);
    statusText.textContent = "Prediction ready.";
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    swapButton.disabled = false;
  }
});

loadTeams().catch((error) => {
  statusText.textContent = error.message;
});

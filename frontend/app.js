/**
 * app.js
 * Frontend controller for Hotel Room Dynamic Pricing Dashboard (INR / Indian Rupees ₹).
 * Handles interactive controls, async fetch to FastAPI backend, and dynamic UI/chart updates.
 */

const API_BASE_URL = (window.location.protocol.startsWith("http") && window.location.port === "8000")
  ? window.location.origin
  : "http://127.0.0.1:8000";

let priceChart = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  checkApiHealth();
  calculateOptimalPrice();
});

/**
 * Currency Formatter helper for Indian Rupees
 */
function formatINR(val, decimals = 2) {
  if (isNaN(val) || val === null) return "₹0.00";
  return "₹" + Number(val).toLocaleString("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

/**
 * Initialize Chart.js Comparison Bar Chart in INR (₹)
 */
function initChart() {
  const ctx = document.getElementById("priceComparisonChart").getContext("2d");
  priceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Baseline Fixed", "Competitor Benchmark", "LP (Network RM)", "DP (Bellman MDP)"],
      datasets: [{
        label: "Room Rate (₹)",
        data: [5500.0, 6500.0, 6500.0, 7800.0],
        backgroundColor: [
          "rgba(100, 116, 139, 0.75)", // Gray (Baseline)
          "rgba(245, 158, 11, 0.75)",  // Amber (Competitor)
          "rgba(6, 182, 212, 0.85)",   // Cyan (LP)
          "rgba(79, 70, 229, 0.9)"     // Indigo (DP)
        ],
        borderColor: [
          "#64748b",
          "#f59e0b",
          "#06b6d4",
          "#4f46e5"
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` Rate: ${formatINR(context.raw, 2)}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: (value) => "₹" + Number(value).toLocaleString("en-IN")
          },
          grid: { color: "rgba(226, 232, 240, 0.6)" }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

/**
 * Check Backend Health & Update Status Badge
 */
async function checkApiHealth() {
  const badge = document.getElementById("apiStatusBadge");
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET" });
    if (res.ok) {
      badge.className = "badge py-2 px-3 rounded-pill badge-connected";
      badge.innerHTML = `<i class="bi bi-check-circle-fill text-success"></i> API Connected (localhost:8000)`;
    } else {
      throw new Error("API returned non-200");
    }
  } catch (err) {
    badge.className = "badge py-2 px-3 rounded-pill badge-disconnected";
    badge.innerHTML = `<i class="bi bi-exclamation-triangle-fill text-danger"></i> API Offline (start server.py)`;
  }
}

/**
 * Two-way Slider & Input Synchronizations
 */
function syncDays(val) {
  val = Math.max(1, Math.min(30, parseInt(val) || 1));
  document.getElementById("daysRange").value = val;
  document.getElementById("daysInput").value = val;
  document.getElementById("daysValueDisplay").innerText = `${val} days`;
}

function syncOccupancy(val) {
  val = Math.max(0, Math.min(100, parseInt(val) || 0));
  document.getElementById("occupancyRange").value = val;
  document.getElementById("occupancyInput").value = val;
  document.getElementById("occupancyValueDisplay").innerText = `${val}%`;
}

function syncCompPrice(val) {
  val = Math.max(1000, parseFloat(val) || 4500);
  document.getElementById("compPriceRange").value = Math.min(25000, val);
  document.getElementById("compPriceInput").value = val.toFixed(2);
  document.getElementById("compPriceValueDisplay").innerText = formatINR(val, 2);
}

/**
 * Apply Scenario Presets (in INR ₹)
 */
function applyPreset(days, occ, compPrice, label) {
  syncDays(days);
  syncOccupancy(occ);
  syncCompPrice(compPrice);
  calculateOptimalPrice();
}

/**
 * Main Async Calculation: Calls POST /optimize-price on FastAPI Backend
 */
async function calculateOptimalPrice() {
  const calcBtn = document.getElementById("calcBtn");
  const origBtnContent = calcBtn.innerHTML;
  calcBtn.disabled = true;
  calcBtn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Solving OR Models...`;

  const daysToCheckin = parseInt(document.getElementById("daysInput").value);
  const occupancyPct = parseFloat(document.getElementById("occupancyInput").value);
  const competitorPrice = parseFloat(document.getElementById("compPriceInput").value);
  const basePrice = parseFloat(document.getElementById("basePriceInput").value) || 5500.0;
  const totalRooms = parseInt(document.getElementById("totalRoomsInput").value) || 100;

  const elasticityRaw = document.getElementById("elasticityInput").value;
  const elasticity = elasticityRaw ? parseFloat(elasticityRaw) : null;

  const demandRateRaw = document.getElementById("demandRateInput").value;
  const baseDemandRate = demandRateRaw ? parseFloat(demandRateRaw) : null;

  const payload = {
    days_to_checkin: daysToCheckin,
    current_occupancy: occupancyPct / 100.0,
    competitor_avg_price: competitorPrice,
    base_price: basePrice,
    total_rooms: totalRooms,
    price_elasticity_factor: elasticity,
    base_demand_rate: baseDemandRate
  };

  try {
    const response = await fetch(`${API_BASE_URL}/optimize-price`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server responded with HTTP ${response.status}`);
    }

    const data = await response.json();
    renderDashboardResults(data);

    const badge = document.getElementById("apiStatusBadge");
    badge.className = "badge py-2 px-3 rounded-pill badge-connected";
    badge.innerHTML = `<i class="bi bi-check-circle-fill text-success"></i> API Connected (localhost:8000)`;

  } catch (error) {
    console.error("Pricing API Error:", error);
    alert(`Could not fetch optimal price from backend:\n${error.message}\n\nPlease ensure your Python backend is running at http://127.0.0.1:8000`);
    const badge = document.getElementById("apiStatusBadge");
    badge.className = "badge py-2 px-3 rounded-pill badge-disconnected";
    badge.innerHTML = `<i class="bi bi-exclamation-triangle-fill text-danger"></i> API Offline (run server.py)`;
  } finally {
    calcBtn.disabled = false;
    calcBtn.innerHTML = origBtnContent;
  }
}

/**
 * Update UI Elements with API Calculation Results in INR (₹)
 */
function renderDashboardResults(data) {
  const base = data.baseline;
  const dp = data.dynamic_programming;
  const lp = data.linear_programming;
  const inp = data.inputs;

  // 1. KPI Cards
  document.getElementById("kpiBaseline").innerText = formatINR(base.price, 0);

  document.getElementById("kpiDpPrice").innerText = formatINR(dp.price, 0);
  const dpDelta = dp.price - base.price;
  const dpSign = dpDelta >= 0 ? "+" : "";
  document.getElementById("kpiDpDelta").innerText = `${dpSign}${formatINR(dpDelta, 0)} vs base`;

  document.getElementById("kpiLpPrice").innerText = formatINR(lp.price, 0);
  const lpDelta = lp.price - base.price;
  const lpSign = lpDelta >= 0 ? "+" : "";
  document.getElementById("kpiLpDelta").innerText = `${lpSign}${formatINR(lpDelta, 0)} vs base`;

  const bestLiftPct = Math.max(dp.revenue_lift_pct, lp.revenue_lift_pct);
  const bestLiftAbs = Math.max(dp.revenue_lift_abs, lp.revenue_lift_abs);
  const liftSign = bestLiftPct >= 0 ? "+" : "";
  document.getElementById("kpiRevLift").innerText = `${liftSign}${bestLiftPct.toFixed(1)}%`;
  document.getElementById("kpiRevLiftAbs").innerText = `${liftSign}${formatINR(bestLiftAbs, 0)}`;

  // Inventory pill
  document.getElementById("inventoryStatePill").innerText = `${inp.unsold_rooms} Rooms Available (${inp.current_occupancy_pct.toFixed(0)}% Booked)`;

  // 2. Update Comparison Chart
  if (priceChart) {
    priceChart.data.datasets[0].data = [
      base.price,
      inp.competitor_avg_price,
      lp.price,
      dp.price
    ];
    priceChart.update();
  }

  // 3. LP Segment Allocation Bars
  const alloc = lp.allocation || { Budget: 0, Standard: 0, Premium: 0 };
  const totalAlloc = (alloc.Budget || 0) + (alloc.Standard || 0) + (alloc.Premium || 0);
  const safeTotal = Math.max(1, totalAlloc);

  const budgetPct = ((alloc.Budget || 0) / safeTotal) * 100;
  const standardPct = ((alloc.Standard || 0) / safeTotal) * 100;
  const premiumPct = ((alloc.Premium || 0) / safeTotal) * 100;

  const budgetRate = lp.segment_prices?.Budget || inp.competitor_avg_price * 0.8;
  const standardRate = lp.segment_prices?.Standard || inp.competitor_avg_price;
  const premiumRate = lp.segment_prices?.Premium || inp.competitor_avg_price * 1.35;

  document.getElementById("allocBudgetBar").style.width = `${budgetPct}%`;
  document.getElementById("allocBudgetVal").innerText = `${alloc.Budget.toFixed(1)} rooms (${formatINR(budgetRate, 0)})`;

  document.getElementById("allocStandardBar").style.width = `${standardPct}%`;
  document.getElementById("allocStandardVal").innerText = `${alloc.Standard.toFixed(1)} rooms (${formatINR(standardRate, 0)})`;

  document.getElementById("allocPremiumBar").style.width = `${premiumPct}%`;
  document.getElementById("allocPremiumVal").innerText = `${alloc.Premium.toFixed(1)} rooms (${formatINR(premiumRate, 0)})`;

  document.getElementById("lpCpiNote").innerText = `Competitor Price Index: ${lp.competitor_price_index.toFixed(3)} (Adheres to 0.88 - 1.25 parity corridor)`;

  // 4. Comparison Table
  document.getElementById("tblBasePrice").innerText = formatINR(base.price, 2);
  document.getElementById("tblDpPrice").innerText = formatINR(dp.price, 2);
  document.getElementById("tblLpPrice").innerText = formatINR(lp.price, 2);

  document.getElementById("tblDpBid").innerHTML = `${formatINR(dp.marginal_bid_price, 2)} <small class="text-muted">(Opportunity Cost ΔV)</small>`;
  document.getElementById("tblLpShadow").innerHTML = `${formatINR(lp.shadow_price_capacity, 2)} <small class="text-muted">(Capacity Shadow μ)</small>`;

  document.getElementById("tblBaseProb").innerText = `${(base.booking_probability * 100).toFixed(1)}%`;
  document.getElementById("tblDpProb").innerText = `${(dp.booking_probability * 100).toFixed(1)}%`;
  document.getElementById("tblLpProb").innerText = `${(lp.booking_probability * 100).toFixed(1)}%`;

  const baseCpi = base.price / inp.competitor_avg_price;
  const dpCpi = dp.price / inp.competitor_avg_price;
  document.getElementById("tblBaseCpi").innerText = baseCpi.toFixed(3);
  document.getElementById("tblDpCpi").innerText = dpCpi.toFixed(3);
  document.getElementById("tblLpCpi").innerText = lp.competitor_price_index.toFixed(3);

  document.getElementById("tblBaseRev").innerText = formatINR(base.projected_revenue, 2);
  document.getElementById("tblDpRev").innerText = `${formatINR(dp.expected_future_revenue, 2)} (+${dp.revenue_lift_pct.toFixed(1)}%)`;
  document.getElementById("tblLpRev").innerText = `${formatINR(lp.projected_revenue, 2)} (+${lp.revenue_lift_pct.toFixed(1)}%)`;

  if (dp.strategy_note) {
    document.getElementById("tblDpNote").innerText = dp.strategy_note;
  }
  if (lp.strategy_note) {
    document.getElementById("tblLpNote").innerText = lp.strategy_note;
  }
}

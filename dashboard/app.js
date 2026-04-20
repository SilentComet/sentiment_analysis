/* ═══════════════════════════════════════════════════════════════════════════
   SENTIMENT ANALYZER — DASHBOARD APPLICATION
   Interactive visualization of DocumentAnalysisResult
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── Sample document ──────────────────────────────────────────────────
    const SAMPLE_DOC = `Q3 Earnings Report — FY2024

Introduction: The quarter began under significant headwinds. Revenue declined by 8%
year-over-year amid challenging macroeconomic conditions, and we have been rightsizing
our workforce to better align costs with current demand levels.

Product Performance: Product A received outstanding reviews from enterprise clients,
with NPS scores reaching an all-time high of 72. Our engineering team has delivered
a record number of features, and customer adoption is accelerating.

Mid-Year Challenges: Oh great — just as we were gaining momentum, supply-chain
disruptions rattled our hardware division. Product A's margins compressed sharply,
and frankly the results were, shall we say, less than inspiring.

Strategic Realignment: We are rationalizing our go-to-market approach and creating
synergies across the enterprise and SMB units. By leveraging our ecosystem and
moving the needle on mission-critical initiatives, we aim to unlock significant value.

Outlook: Going forward, we expect robust tailwinds from our AI product line.
In the next two quarters, we project revenue growth of 15–20%. The ball is in our
court to capitalize on these opportunities, and we are confident in our ability
to deliver exceptional results. Product A, despite earlier challenges, is now
positioned as a market leader with an outstanding roadmap ahead.`;

    // ── Sample result (pre-computed for demo without backend) ────────────
    const SAMPLE_RESULT = {
        document_metadata: {
            document_id: "demo-001",
            filename: "sample_earnings.txt",
            total_pages: 1,
            total_chunks: 5,
            total_tokens: 478,
            word_count: 189,
            language: "en",
            analyzed_at: new Date().toISOString(),
            processing_time_ms: 342.7,
            inference_route_distribution: { slm: 3, llm: 2 }
        },
        overall_sentiment_score: 0.1842,
        overall_sentiment_label: "neutral",
        sentiment_trajectory: {
            scores: [-0.35, 0.62, -0.28, -0.15, 0.48],
            rolling_mean: [-0.35, 0.14, -0.003, 0.06, 0.17],
            segments: [
                { label: "introduction", chunk_sequence_start: 0, chunk_sequence_end: 0, mean_sentiment: -0.35, trend: "stable", peak_score: -0.35, trough_score: -0.35 },
                { label: "body", chunk_sequence_start: 1, chunk_sequence_end: 3, mean_sentiment: 0.063, trend: "falling", peak_score: 0.62, trough_score: -0.28 },
                { label: "conclusion", chunk_sequence_start: 4, chunk_sequence_end: 4, mean_sentiment: 0.48, trend: "stable", peak_score: 0.48, trough_score: 0.48 }
            ],
            overall_trend: "improving",
            inflection_points: [1, 2, 4],
            intro_sentiment: -0.35,
            conclusion_sentiment: 0.48,
            sentiment_delta: 0.83
        },
        aspect_analysis: [
            {
                entity_text: "product a",
                entity_type: "PRODUCT",
                mention_count: 3,
                first_chunk_sequence: 1,
                last_chunk_sequence: 4,
                aspect_sentiment_score: 0.38,
                aspect_sentiment_label: "positive",
                sentiment_trajectory: [0.62, -0.28, 0.48],
                contradictions: [
                    { earlier_chunk_id: "chunk_00001", later_chunk_id: "chunk_00002", earlier_sentiment_score: 0.62, later_sentiment_score: -0.28, delta: 0.9, resolution_strategy: "recency", resolved_score: -0.28 }
                ],
                chain_of_evidence: {
                    claim_summary: "Sentiment toward 'product a' is positive.",
                    supporting_quotes: [{ quote: "Product A received outstanding reviews from enterprise clients", chunk_id: "chunk_00001", chunk_sequence: 1, relevance_score: 0.92 }],
                    reasoning_steps: ["Claim: positive sentiment", "3 mentions across 3 chunks", "Coverage: 78%"],
                    hallucination_risk_score: 0.12,
                    grounding_coverage: 0.78
                },
                sub_aspects: []
            },
            {
                entity_text: "ai product line",
                entity_type: "PRODUCT",
                mention_count: 1,
                first_chunk_sequence: 4,
                last_chunk_sequence: 4,
                aspect_sentiment_score: 0.55,
                aspect_sentiment_label: "positive",
                sentiment_trajectory: [0.55],
                contradictions: [],
                chain_of_evidence: {
                    claim_summary: "Sentiment toward 'ai product line' is positive.",
                    supporting_quotes: [{ quote: "we expect robust tailwinds from our AI product line", chunk_id: "chunk_00004", chunk_sequence: 4, relevance_score: 0.88 }],
                    reasoning_steps: ["Claim: positive sentiment", "1 mention", "Coverage: 85%"],
                    hallucination_risk_score: 0.08,
                    grounding_coverage: 0.85
                },
                sub_aspects: []
            }
        ],
        emotion_profile: {
            dominant_emotion: "anticipation",
            emotion_distribution: {
                joy: 0.14, sadness: 0.08, anger: 0.03, fear: 0.11,
                surprise: 0.04, disgust: 0.02, anticipation: 0.35, trust: 0.18, neutral: 0.05
            }
        },
        intent_classification: {
            primary_intent: "forecast",
            secondary_intents: ["deflect", "persuade"],
            confidence: 0.82,
            implicit_intent_notes: "Document uses forward-looking language and hedges past performance with external factors."
        },
        detected_idioms: ["the ball is in your court"],
        detected_corporate_speak: ["headwinds", "rightsizing", "synergies", "ecosystem", "mission-critical", "move the needle", "tailwinds", "leverage"],
        sarcasm_detected: true,
        document_level_evidence: {
            claim_summary: "The document has an overall neutral sentiment (score: 0.184), improving trajectory.",
            supporting_quotes: [
                { quote: "Revenue declined by 8% year-over-year amid challenging macroeconomic conditions", chunk_id: "chunk_00000", chunk_sequence: 0, relevance_score: 0.85 },
                { quote: "we are confident in our ability to deliver exceptional results", chunk_id: "chunk_00004", chunk_sequence: 4, relevance_score: 0.90 }
            ],
            reasoning_steps: [
                "Claim: 'The document has an overall neutral sentiment'",
                "Identified 2 supporting passages across 5 chunks.",
                "Token-level grounding coverage: 72%.",
                "Hallucination risk score: 0.15 (LOW)."
            ],
            hallucination_risk_score: 0.15,
            grounding_coverage: 0.72
        },
        chunk_results: [
            { chunk_id: "chunk_00000", chunk_sequence: 0, page_number: 1, text_preview: "Q3 Earnings Report — FY2024 Introduction: The quarter began under significant headwinds. Revenue declined by 8% year-over-year amid challenging macroecono...", sentiment_score: -0.35, sentiment_label: "negative", dominant_emotion: "fear", emotion_scores: { fear: 0.3, sadness: 0.25, anticipation: 0.15, trust: 0.1, neutral: 0.2 }, complexity_score: 0.42, inference_route: "slm", inference_latency_ms: 28.4, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: true, cultural_adjustments: ["Corporate speak 'headwinds' decoded as: obstacles / challenges", "Corporate speak 'rightsizing' decoded as: layoffs"], confidence: 0.78 },
            { chunk_id: "chunk_00001", chunk_sequence: 1, page_number: 1, text_preview: "Product Performance: Product A received outstanding reviews from enterprise clients, with NPS scores reaching an all-time high of 72. Our engineering team...", sentiment_score: 0.62, sentiment_label: "very_positive", dominant_emotion: "joy", emotion_scores: { joy: 0.45, trust: 0.25, anticipation: 0.15, neutral: 0.15 }, complexity_score: 0.18, inference_route: "slm", inference_latency_ms: 22.1, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: false, cultural_adjustments: [], confidence: 0.91 },
            { chunk_id: "chunk_00002", chunk_sequence: 2, page_number: 1, text_preview: "Mid-Year Challenges: Oh great — just as we were gaining momentum, supply-chain disruptions rattled our hardware division. Product A's margins compressed s...", sentiment_score: -0.28, sentiment_label: "negative", dominant_emotion: "disgust", emotion_scores: { disgust: 0.25, anger: 0.2, sadness: 0.2, surprise: 0.15, neutral: 0.2 }, complexity_score: 0.72, inference_route: "llm", inference_latency_ms: 1284.3, has_sarcasm_signal: true, has_idiom_signal: false, has_corporate_speak: false, cultural_adjustments: [], confidence: 0.85 },
            { chunk_id: "chunk_00003", chunk_sequence: 3, page_number: 1, text_preview: "Strategic Realignment: We are rationalizing our go-to-market approach and creating synergies across the enterprise and SMB units. By leveraging our ecosys...", sentiment_score: -0.15, sentiment_label: "neutral", dominant_emotion: "anticipation", emotion_scores: { anticipation: 0.35, trust: 0.2, neutral: 0.25, fear: 0.1, joy: 0.1 }, complexity_score: 0.65, inference_route: "llm", inference_latency_ms: 1156.7, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: true, cultural_adjustments: ["Corporate speak 'synergies' decoded as: cost-cutting via merger", "Corporate speak 'ecosystem' decoded as: platform with lock-in", "Corporate speak 'mission-critical' decoded as: very important", "Corporate speak 'move the needle' decoded as: make measurable progress", "Corporate speak 'leverage' decoded as: use strategically"], confidence: 0.79 },
            { chunk_id: "chunk_00004", chunk_sequence: 4, page_number: 1, text_preview: "Outlook: Going forward, we expect robust tailwinds from our AI product line. In the next two quarters, we project revenue growth of 15–20%. The ball is i...", sentiment_score: 0.48, sentiment_label: "positive", dominant_emotion: "anticipation", emotion_scores: { anticipation: 0.4, joy: 0.2, trust: 0.25, neutral: 0.15 }, complexity_score: 0.38, inference_route: "slm", inference_latency_ms: 25.8, has_sarcasm_signal: false, has_idiom_signal: true, has_corporate_speak: true, cultural_adjustments: ["Corporate speak 'tailwinds' decoded as: favourable conditions", "Idiom 'the ball is in your court' decoded as: it is your responsibility now", "Implicit intent detected: forecast"], confidence: 0.87 }
        ],
        confidence_metrics: {
            overall_confidence: 0.82,
            chunk_coverage: 1.0,
            evidence_density: 0.72,
            model_agreement_score: null,
            sarcasm_detection_confidence: 0.90,
            uncertainty_flags: []
        }
    };

    // ── DOM refs ──────────────────────────────────────────────────────────
    const $  = id => document.getElementById(id);
    const docInput     = $("doc-input");
    const btnAnalyze   = $("btn-analyze");
    const btnSample    = $("btn-sample");
    const btnClear     = $("btn-clear");
    const progressBar  = $("progress-bar");
    const resultsCont  = $("results-container");
    const dropZone     = $("drop-zone");
    const badgeStatus  = $("badge-status");

    // ── Event Listeners ──────────────────────────────────────────────────
    docInput.addEventListener("input", () => {
        btnAnalyze.disabled = docInput.value.trim().length < 20;
    });

    btnSample.addEventListener("click", () => {
        docInput.value = SAMPLE_DOC;
        btnAnalyze.disabled = false;
        docInput.dispatchEvent(new Event("input"));
    });

    btnClear.addEventListener("click", () => {
        docInput.value = "";
        btnAnalyze.disabled = true;
        resultsCont.hidden = true;
    });

    // Drag & drop
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = ev => { docInput.value = ev.target.result; btnAnalyze.disabled = false; };
            reader.readAsText(file);
        }
    });

    // Analyze button
    btnAnalyze.addEventListener("click", () => runAnalysis());

    $("btn-toggle-json").addEventListener("click", () => {
        const pre = $("raw-json");
        pre.hidden = !pre.hidden;
        $("btn-toggle-json").textContent = pre.hidden ? "Show Raw JSON" : "Hide Raw JSON";
    });

    // ── Analysis pipeline ────────────────────────────────────────────────
    async function runAnalysis() {
        const btnText = btnAnalyze.querySelector(".btn-text");
        const btnLoader = btnAnalyze.querySelector(".btn-loader");
        btnText.textContent = "Analyzing…";
        btnLoader.hidden = false;
        btnAnalyze.disabled = true;
        badgeStatus.textContent = "Processing";
        badgeStatus.className = "badge badge-purple";
        progressBar.hidden = false;

        const fill = progressBar.querySelector(".progress-fill");
        let pct = 0;
        const ticker = setInterval(() => {
            pct = Math.min(pct + Math.random() * 12, 92);
            fill.style.width = pct + "%";
        }, 200);

        // Simulate async processing (in production, POST to API)
        await new Promise(r => setTimeout(r, 1800));

        clearInterval(ticker);
        fill.style.width = "100%";

        setTimeout(() => {
            renderResults(SAMPLE_RESULT);
            btnText.textContent = "Analyze Document";
            btnLoader.hidden = true;
            btnAnalyze.disabled = false;
            badgeStatus.textContent = "Complete";
            badgeStatus.className = "badge badge-blue";
            progressBar.hidden = true;
            fill.style.width = "0%";
            resultsCont.hidden = false;
            resultsCont.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 400);
    }

    // ── Render engine ────────────────────────────────────────────────────
    function renderResults(data) {
        renderGauge(data.overall_sentiment_score, data.overall_sentiment_label);
        renderScorePills(data);
        renderMetadata(data.document_metadata);
        renderTrajectory(data.sentiment_trajectory);
        renderEmotion(data.emotion_profile);
        renderAspects(data.aspect_analysis);
        renderEvidence(data.document_level_evidence);
        renderIntent(data.intent_classification, data.detected_idioms, data.detected_corporate_speak, data.sarcasm_detected);
        renderChunks(data.chunk_results);
        renderConfidence(data.confidence_metrics);
        $("raw-json").textContent = JSON.stringify(data, null, 2);
    }

    // ── Gauge ────────────────────────────────────────────────────────────
    function renderGauge(score, label) {
        const canvas = $("gauge-canvas");
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const cx = W / 2, cy = H - 10, radius = 100;
        const startAngle = Math.PI, endAngle = 2 * Math.PI;

        // Background arc
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, endAngle);
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 14;
        ctx.lineCap = "round";
        ctx.stroke();

        // Gradient arc
        const normalized = (score + 1) / 2; // 0..1
        const grad = ctx.createLinearGradient(cx - radius, 0, cx + radius, 0);
        grad.addColorStop(0, "#ef4444");
        grad.addColorStop(0.35, "#fbbf24");
        grad.addColorStop(0.5, "#38bdf8");
        grad.addColorStop(0.7, "#4ade80");
        grad.addColorStop(1, "#22c55e");

        const targetAngle = startAngle + normalized * Math.PI;
        // Animate
        let currentAngle = startAngle;
        function animateGauge() {
            currentAngle += (targetAngle - currentAngle) * 0.08;
            ctx.clearRect(0, 0, W, H);

            // BG arc
            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, endAngle);
            ctx.strokeStyle = "rgba(255,255,255,0.06)";
            ctx.lineWidth = 14;
            ctx.lineCap = "round";
            ctx.stroke();

            // Value arc
            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, currentAngle);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 14;
            ctx.lineCap = "round";
            ctx.stroke();

            // Tick marks
            for (let i = 0; i <= 10; i++) {
                const a = startAngle + (i / 10) * Math.PI;
                const inner = radius - 22;
                const outer = radius - 16;
                ctx.beginPath();
                ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
                ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer);
                ctx.strokeStyle = "rgba(255,255,255,0.1)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            if (Math.abs(currentAngle - targetAngle) > 0.005) {
                requestAnimationFrame(animateGauge);
            }
        }
        animateGauge();

        // Animate counter
        const valueEl = $("gauge-value");
        let currentVal = 0;
        function animateCounter() {
            currentVal += (score - currentVal) * 0.06;
            valueEl.textContent = currentVal.toFixed(2);
            if (Math.abs(currentVal - score) > 0.005) requestAnimationFrame(animateCounter);
            else valueEl.textContent = score.toFixed(4);
        }
        animateCounter();

        $("gauge-label").textContent = label.replace(/_/g, " ");
    }

    // ── Score pills ──────────────────────────────────────────────────────
    function renderScorePills(data) {
        const cont = $("score-pills");
        const traj = data.sentiment_trajectory;
        const pills = [
            { label: `Δ ${traj.sentiment_delta > 0 ? "+" : ""}${traj.sentiment_delta.toFixed(2)}`, cls: traj.sentiment_delta >= 0 ? "pill-pos" : "pill-neg" },
            { label: `Arc: ${traj.overall_trend}`, cls: "pill-neu" },
            { label: data.sarcasm_detected ? "⚡ Sarcasm" : "No Sarcasm", cls: data.sarcasm_detected ? "pill-mixed" : "pill-neu" },
            { label: `${data.chunk_results.length} chunks`, cls: "pill-neu" },
        ];
        cont.innerHTML = pills.map(p => `<span class="pill ${p.cls}">${p.label}</span>`).join("");
    }

    // ── Metadata ─────────────────────────────────────────────────────────
    function renderMetadata(meta) {
        const items = [
            { label: "Pages", value: meta.total_pages },
            { label: "Chunks", value: meta.total_chunks },
            { label: "Tokens", value: meta.total_tokens.toLocaleString() },
            { label: "Words", value: meta.word_count.toLocaleString() },
            { label: "Language", value: meta.language.toUpperCase() },
            { label: "Processing", value: `${meta.processing_time_ms.toFixed(0)}ms` },
            { label: "SLM Chunks", value: meta.inference_route_distribution.slm || 0 },
            { label: "LLM Chunks", value: meta.inference_route_distribution.llm || 0 },
        ];
        $("meta-grid").innerHTML = items.map(i => `
            <div class="meta-item">
                <div class="meta-label">${i.label}</div>
                <div class="meta-value">${i.value}</div>
            </div>
        `).join("");
    }

    // ── Trajectory chart ─────────────────────────────────────────────────
    function renderTrajectory(traj) {
        $("traj-legend").innerHTML = `<span class="legend-raw">Raw Scores</span><span class="legend-smooth">Rolling Mean</span>`;

        const canvas = $("trajectory-canvas");
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = 440;
        const ctx = canvas.getContext("2d");
        ctx.scale(2, 2);

        const W = canvas.offsetWidth, H = 220;
        const padL = 40, padR = 20, padT = 20, padB = 30;
        const plotW = W - padL - padR, plotH = H - padT - padB;
        const n = traj.scores.length;
        if (n === 0) return;

        // Grid
        ctx.strokeStyle = "rgba(255,255,255,0.04)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padT + (i / 4) * plotH;
            ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        }

        // Zero line
        const zeroY = padT + plotH / 2;
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();
        ctx.setLineDash([]);

        // Y-axis labels
        ctx.fillStyle = "rgba(255,255,255,0.3)";
        ctx.font = "10px 'JetBrains Mono'";
        ctx.textAlign = "right";
        ctx.fillText("+1.0", padL - 6, padT + 4);
        ctx.fillText(" 0.0", padL - 6, zeroY + 4);
        ctx.fillText("-1.0", padL - 6, padT + plotH + 4);

        function toX(i) { return padL + (i / Math.max(n - 1, 1)) * plotW; }
        function toY(v) { return padT + ((1 - v) / 2) * plotH; }

        // Segment backgrounds
        const segColors = { introduction: "rgba(56,189,248,0.04)", body: "rgba(167,139,250,0.04)", conclusion: "rgba(74,222,128,0.04)" };
        traj.segments.forEach(seg => {
            const x1 = toX(seg.chunk_sequence_start);
            const x2 = toX(seg.chunk_sequence_end);
            ctx.fillStyle = segColors[seg.label] || "rgba(255,255,255,0.02)";
            ctx.fillRect(x1, padT, x2 - x1 + (plotW / n), plotH);
            ctx.fillStyle = "rgba(255,255,255,0.15)";
            ctx.font = "9px Inter";
            ctx.textAlign = "center";
            ctx.fillText(seg.label, (x1 + x2) / 2 + (plotW / n / 2), H - 6);
        });

        // Rolling mean area fill
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(traj.rolling_mean[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(toX(i), toY(traj.rolling_mean[i]));
        ctx.lineTo(toX(n - 1), zeroY);
        ctx.lineTo(toX(0), zeroY);
        ctx.closePath();
        const areaGrad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
        areaGrad.addColorStop(0, "rgba(56,189,248,0.08)");
        areaGrad.addColorStop(1, "rgba(56,189,248,0.0)");
        ctx.fillStyle = areaGrad;
        ctx.fill();

        // Rolling mean line
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(traj.rolling_mean[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(toX(i), toY(traj.rolling_mean[i]));
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Raw scores line
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(traj.scores[0]));
        for (let i = 1; i < n; i++) ctx.lineTo(toX(i), toY(traj.scores[i]));
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Data points
        traj.scores.forEach((s, i) => {
            ctx.beginPath();
            ctx.arc(toX(i), toY(s), 4, 0, Math.PI * 2);
            ctx.fillStyle = s >= 0 ? "#4ade80" : "#fb7185";
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.4)";
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        // Inflection markers
        traj.inflection_points.forEach(ip => {
            if (ip < n) {
                ctx.beginPath();
                ctx.arc(toX(ip), toY(traj.scores[ip]), 8, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(251,191,36,0.5)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });

        // Arc badges
        const badges = $("arc-badges");
        badges.innerHTML = [
            `Trend: ${traj.overall_trend}`,
            `Intro: ${traj.intro_sentiment.toFixed(2)}`,
            `Conclusion: ${traj.conclusion_sentiment.toFixed(2)}`,
            `Δ: ${traj.sentiment_delta > 0 ? "+" : ""}${traj.sentiment_delta.toFixed(2)}`,
            `${traj.inflection_points.length} inflections`,
        ].map(t => `<span class="arc-badge">${t}</span>`).join("");
    }

    // ── Emotion radar chart ──────────────────────────────────────────────
    function renderEmotion(profile) {
        const canvas = $("emotion-canvas");
        const ctx = canvas.getContext("2d");
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const emotions = Object.entries(profile.emotion_distribution).filter(([k]) => k !== "neutral");
        const n = emotions.length;
        if (n === 0) return;

        const cx = W / 2, cy = H / 2, maxR = 110;
        const angleStep = (Math.PI * 2) / n;

        // Grid circles
        for (let r = 1; r <= 4; r++) {
            ctx.beginPath();
            ctx.arc(cx, cy, (r / 4) * maxR, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Grid lines
        emotions.forEach((_, i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(angle) * maxR, cy + Math.sin(angle) * maxR);
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.stroke();
        });

        // Data polygon
        const maxVal = Math.max(...emotions.map(([, v]) => v), 0.01);
        ctx.beginPath();
        emotions.forEach(([, val], i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            const r = (val / maxVal) * maxR * 0.9;
            const x = cx + Math.cos(angle) * r;
            const y = cy + Math.sin(angle) * r;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();

        const polyGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
        polyGrad.addColorStop(0, "rgba(167,139,250,0.3)");
        polyGrad.addColorStop(1, "rgba(56,189,248,0.1)");
        ctx.fillStyle = polyGrad;
        ctx.fill();
        ctx.strokeStyle = "rgba(167,139,250,0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Labels
        const emoColors = {
            joy: "#4ade80", sadness: "#38bdf8", anger: "#ef4444", fear: "#fbbf24",
            surprise: "#a78bfa", disgust: "#fb7185", anticipation: "#2dd4bf", trust: "#38bdf8"
        };
        ctx.font = "11px Inter";
        ctx.textAlign = "center";
        emotions.forEach(([label, val], i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            const lx = cx + Math.cos(angle) * (maxR + 18);
            const ly = cy + Math.sin(angle) * (maxR + 18);
            ctx.fillStyle = emoColors[label] || "rgba(255,255,255,0.5)";
            ctx.fillText(label, lx, ly + 4);

            // Data point
            const r = (val / maxVal) * maxR * 0.9;
            const dx = cx + Math.cos(angle) * r;
            const dy = cy + Math.sin(angle) * r;
            ctx.beginPath();
            ctx.arc(dx, dy, 4, 0, Math.PI * 2);
            ctx.fillStyle = emoColors[label] || "#a78bfa";
            ctx.fill();
        });

        $("emotion-dominant").innerHTML = `Dominant: <span class="text-purple" style="text-transform:capitalize">${profile.dominant_emotion}</span>`;
    }

    // ── Aspects ──────────────────────────────────────────────────────────
    function renderAspects(aspects) {
        const grid = $("aspects-grid");
        if (!aspects.length) { grid.innerHTML = "<p style='color:var(--text-muted)'>No entities detected.</p>"; return; }

        grid.innerHTML = aspects.map(a => {
            const scoreColor = a.aspect_sentiment_score >= 0.2 ? "text-green" : a.aspect_sentiment_score <= -0.2 ? "text-red" : "text-blue";
            const barPct = ((a.aspect_sentiment_score + 1) / 2 * 100).toFixed(1);
            const barColor = a.aspect_sentiment_score >= 0.2 ? "var(--accent-green)" : a.aspect_sentiment_score <= -0.2 ? "var(--accent-red)" : "var(--accent-blue)";
            const trajBars = a.sentiment_trajectory.map(s => {
                const h = Math.max(2, Math.abs(s) * 22);
                const c = s >= 0 ? "var(--accent-green)" : "var(--accent-rose)";
                return `<div class="traj-bar" style="height:${h}px;background:${c}"></div>`;
            }).join("");

            return `
                <div class="aspect-card">
                    <div class="aspect-header">
                        <span class="aspect-name">${a.entity_text}</span>
                        <span class="aspect-type">${a.entity_type}</span>
                    </div>
                    <div class="aspect-score ${scoreColor}">${a.aspect_sentiment_score.toFixed(4)}</div>
                    <div class="aspect-bar"><div class="aspect-bar-fill" style="width:${barPct}%;background:${barColor}"></div></div>
                    <div class="aspect-mentions">${a.mention_count} mention${a.mention_count > 1 ? "s" : ""} · ${a.contradictions.length} contradiction${a.contradictions.length !== 1 ? "s" : ""}</div>
                    <div class="aspect-trajectory-mini">${trajBars}</div>
                </div>
            `;
        }).join("");
    }

    // ── Evidence ─────────────────────────────────────────────────────────
    function renderEvidence(coe) {
        const riskCls = coe.hallucination_risk_score < 0.3 ? "risk-low" : coe.hallucination_risk_score < 0.6 ? "risk-medium" : "risk-high";
        const riskLabel = coe.hallucination_risk_score < 0.3 ? "LOW RISK" : coe.hallucination_risk_score < 0.6 ? "MEDIUM RISK" : "HIGH RISK";

        $("evidence-content").innerHTML = `
            <p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:var(--space-md)">${coe.claim_summary}</p>
            ${coe.supporting_quotes.map(q => `
                <div class="evidence-item">
                    <div class="evidence-quote">${q.quote}</div>
                    <div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px">${q.chunk_id} · relevance: ${(q.relevance_score * 100).toFixed(0)}%</div>
                </div>
            `).join("")}
            <div style="margin-top:var(--space-md);display:flex;gap:var(--space-sm);align-items:center">
                <span class="risk-badge ${riskCls}">Hallucination: ${riskLabel} (${coe.hallucination_risk_score.toFixed(2)})</span>
                <span class="risk-badge risk-low">Coverage: ${(coe.grounding_coverage * 100).toFixed(0)}%</span>
            </div>
            <div style="margin-top:var(--space-md)">
                <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:4px">Reasoning Steps:</div>
                <ol style="padding-left:var(--space-lg);font-size:0.78rem;color:var(--text-secondary)">
                    ${coe.reasoning_steps.map(s => `<li>${s}</li>`).join("")}
                </ol>
            </div>
        `;
    }

    // ── Intent & Signals ─────────────────────────────────────────────────
    function renderIntent(intent, idioms, corpSpeak, sarcasm) {
        $("intent-content").innerHTML = `
            <div class="intent-block">
                <div class="intent-label">Primary Intent</div>
                <div style="font-size:1.1rem;font-weight:800;text-transform:capitalize;color:var(--accent-teal)">${intent.primary_intent}</div>
                <div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">Confidence: ${(intent.confidence * 100).toFixed(0)}%</div>
                ${intent.secondary_intents.length ? `<div style="font-size:0.72rem;color:var(--text-muted);margin-top:2px">Secondary: ${intent.secondary_intents.join(", ")}</div>` : ""}
                ${intent.implicit_intent_notes ? `<div style="font-size:0.78rem;color:var(--text-secondary);margin-top:var(--space-sm);font-style:italic">${intent.implicit_intent_notes}</div>` : ""}
            </div>
            ${corpSpeak.length ? `
                <div class="intent-block">
                    <div class="intent-label">Corporate Speak Detected</div>
                    <ul class="signal-list">${corpSpeak.map(p => `<li>${p}</li>`).join("")}</ul>
                </div>
            ` : ""}
            ${idioms.length ? `
                <div class="intent-block">
                    <div class="intent-label">Idioms Detected</div>
                    <ul class="signal-list">${idioms.map(p => `<li>${p}</li>`).join("")}</ul>
                </div>
            ` : ""}
            ${sarcasm ? `<div class="intent-block"><div class="intent-label" style="color:var(--accent-amber)">⚡ Sarcasm Detected</div></div>` : ""}
        `;
    }

    // ── Chunks accordion ─────────────────────────────────────────────────
    function renderChunks(chunks) {
        $("chunk-count").textContent = `(${chunks.length})`;
        const cont = $("chunks-accordion");
        cont.innerHTML = chunks.map(c => {
            const barPct = ((c.sentiment_score + 1) / 2 * 100).toFixed(1);
            const barColor = c.sentiment_score >= 0.2 ? "var(--accent-green)" : c.sentiment_score <= -0.2 ? "var(--accent-rose)" : "var(--accent-blue)";
            const scoreColor = c.sentiment_score >= 0.2 ? "text-green" : c.sentiment_score <= -0.2 ? "text-red" : "text-blue";
            const routeCls = c.inference_route === "slm" ? "route-slm" : c.inference_route === "llm" ? "route-llm" : "route-cached";

            return `
                <div class="chunk-item" data-chunk="${c.chunk_id}">
                    <div class="chunk-header" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span class="chunk-id">${c.chunk_id}</span>
                        <div class="chunk-score-bar"><div class="chunk-score-fill" style="width:${barPct}%;background:${barColor}"></div></div>
                        <span class="chunk-score-val ${scoreColor}">${c.sentiment_score.toFixed(2)}</span>
                        <span class="chunk-route-badge ${routeCls}">${c.inference_route}</span>
                    </div>
                    <div class="chunk-body">
                        <p style="margin-bottom:var(--space-sm)">${c.text_preview}</p>
                        <div class="chunk-detail-grid">
                            <div class="chunk-detail"><div class="chunk-detail-label">Label</div><div class="chunk-detail-value">${c.sentiment_label}</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Emotion</div><div class="chunk-detail-value">${c.dominant_emotion}</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Complexity</div><div class="chunk-detail-value">${c.complexity_score.toFixed(2)}</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Latency</div><div class="chunk-detail-value">${c.inference_latency_ms.toFixed(0)}ms</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Confidence</div><div class="chunk-detail-value">${(c.confidence * 100).toFixed(0)}%</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Sarcasm</div><div class="chunk-detail-value">${c.has_sarcasm_signal ? "⚡ Yes" : "No"}</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Corp. Speak</div><div class="chunk-detail-value">${c.has_corporate_speak ? "Yes" : "No"}</div></div>
                            <div class="chunk-detail"><div class="chunk-detail-label">Idiom</div><div class="chunk-detail-value">${c.has_idiom_signal ? "Yes" : "No"}</div></div>
                        </div>
                        ${c.cultural_adjustments.length ? `
                            <div style="margin-top:var(--space-sm)">
                                <div class="chunk-detail-label" style="margin-bottom:4px">Cultural Adjustments</div>
                                <ul class="signal-list">${c.cultural_adjustments.map(a => `<li>${a}</li>`).join("")}</ul>
                            </div>
                        ` : ""}
                    </div>
                </div>
            `;
        }).join("");
    }

    // ── Confidence metrics ───────────────────────────────────────────────
    function renderConfidence(metrics) {
        const items = [
            { label: "Overall Confidence", value: metrics.overall_confidence },
            { label: "Chunk Coverage", value: metrics.chunk_coverage },
            { label: "Evidence Density", value: metrics.evidence_density },
            { label: "Sarcasm Detection", value: metrics.sarcasm_detection_confidence },
        ];
        const cont = $("confidence-bars");
        cont.innerHTML = items.map(i => `
            <div class="conf-item">
                <div class="conf-label"><span>${i.label}</span><span>${(i.value * 100).toFixed(0)}%</span></div>
                <div class="conf-bar"><div class="conf-fill" style="width:${(i.value * 100).toFixed(1)}%"></div></div>
            </div>
        `).join("");

        if (metrics.uncertainty_flags.length) {
            cont.innerHTML += `
                <div class="uncertainty-flags" style="grid-column: 1/-1">
                    ${metrics.uncertainty_flags.map(f => `<span class="flag-tag">⚠ ${f.replace(/_/g, " ")}</span>`).join("")}
                </div>
            `;
        }
    }

})();

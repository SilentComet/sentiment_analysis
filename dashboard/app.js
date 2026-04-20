/* ═══════════════════════════════════════════════════════════════════════════
   SENTIMENT ANALYZER — OBSIDIAN DASHBOARD
   Spring-physics gauge · Catmull-Rom trajectory · Staggered reveals
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── CSS variable reader ──────────────────────────────────────────────
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

    // ── Sample document ──────────────────────────────────────────────────
    const SAMPLE_DOC = `Q3 Earnings Report — FY2024

Introduction: The quarter began under significant headwinds. Revenue declined by 8% year-over-year amid challenging macroeconomic conditions, and we have been rightsizing our workforce to better align costs with current demand levels.

Product Performance: Product A received outstanding reviews from enterprise clients, with NPS scores reaching an all-time high of 72. Our engineering team has delivered a record number of features, and customer adoption is accelerating.

Mid-Year Challenges: Oh great — just as we were gaining momentum, supply-chain disruptions rattled our hardware division. Product A's margins compressed sharply, and frankly the results were, shall we say, less than inspiring.

Strategic Realignment: We are rationalizing our go-to-market approach and creating synergies across the enterprise and SMB units. By leveraging our ecosystem and moving the needle on mission-critical initiatives, we aim to unlock significant value.

Outlook: Going forward, we expect robust tailwinds from our AI product line. In the next two quarters, we project revenue growth of 15–20%. The ball is in our court to capitalize on these opportunities, and we are confident in our ability to deliver exceptional results. Product A, despite earlier challenges, is now positioned as a market leader with an outstanding roadmap ahead.`;

    // ── Demo result ──────────────────────────────────────────────────────
    const DEMO = {
        document_metadata: { document_id: "demo-001", filename: "sample_earnings.txt", total_pages: 1, total_chunks: 5, total_tokens: 478, word_count: 189, language: "en", analyzed_at: new Date().toISOString(), processing_time_ms: 342.7, inference_route_distribution: { slm: 3, llm: 2 } },
        overall_sentiment_score: 0.1842,
        overall_sentiment_label: "neutral",
        sentiment_trajectory: {
            scores: [-0.35, 0.62, -0.28, -0.15, 0.48],
            rolling_mean: [-0.35, 0.14, -0.003, 0.06, 0.17],
            segments: [
                { label: "intro", chunk_sequence_start: 0, chunk_sequence_end: 0, mean_sentiment: -0.35, trend: "stable", peak_score: -0.35, trough_score: -0.35 },
                { label: "body", chunk_sequence_start: 1, chunk_sequence_end: 3, mean_sentiment: 0.063, trend: "falling", peak_score: 0.62, trough_score: -0.28 },
                { label: "conclusion", chunk_sequence_start: 4, chunk_sequence_end: 4, mean_sentiment: 0.48, trend: "stable", peak_score: 0.48, trough_score: 0.48 }
            ],
            overall_trend: "improving", inflection_points: [1, 2, 4],
            intro_sentiment: -0.35, conclusion_sentiment: 0.48, sentiment_delta: 0.83
        },
        aspect_analysis: [
            { entity_text: "product a", entity_type: "PRODUCT", mention_count: 3, first_chunk_sequence: 1, last_chunk_sequence: 4, aspect_sentiment_score: 0.38, aspect_sentiment_label: "positive", sentiment_trajectory: [0.62, -0.28, 0.48], contradictions: [{ earlier_chunk_id: "c1", later_chunk_id: "c2", earlier_sentiment_score: 0.62, later_sentiment_score: -0.28, delta: 0.9, resolution_strategy: "recency", resolved_score: -0.28 }], chain_of_evidence: { claim_summary: "Sentiment toward 'product a' is positive.", supporting_quotes: [{ quote: "Product A received outstanding reviews from enterprise clients", chunk_id: "chunk_00001", chunk_sequence: 1, relevance_score: 0.92 }], reasoning_steps: ["Positive entity", "3 mentions", "Coverage: 78%"], hallucination_risk_score: 0.12, grounding_coverage: 0.78 }, sub_aspects: [] },
            { entity_text: "ai product line", entity_type: "PRODUCT", mention_count: 1, first_chunk_sequence: 4, last_chunk_sequence: 4, aspect_sentiment_score: 0.55, aspect_sentiment_label: "positive", sentiment_trajectory: [0.55], contradictions: [], chain_of_evidence: { claim_summary: "Sentiment toward 'ai product line' is positive.", supporting_quotes: [{ quote: "we expect robust tailwinds from our AI product line", chunk_id: "chunk_00004", chunk_sequence: 4, relevance_score: 0.88 }], reasoning_steps: ["Positive forecast", "1 mention", "Coverage: 85%"], hallucination_risk_score: 0.08, grounding_coverage: 0.85 }, sub_aspects: [] }
        ],
        emotion_profile: { dominant_emotion: "anticipation", emotion_distribution: { joy: 0.14, sadness: 0.08, anger: 0.03, fear: 0.11, surprise: 0.04, disgust: 0.02, anticipation: 0.35, trust: 0.18, neutral: 0.05 } },
        intent_classification: { primary_intent: "forecast", secondary_intents: ["deflect", "persuade"], confidence: 0.82, implicit_intent_notes: "Forward-looking language hedges past performance with external factors." },
        detected_idioms: ["the ball is in your court"],
        detected_corporate_speak: ["headwinds", "rightsizing", "synergies", "ecosystem", "mission-critical", "move the needle", "tailwinds", "leverage"],
        sarcasm_detected: true,
        document_level_evidence: { claim_summary: "The document has an overall neutral sentiment (score: 0.184), improving trajectory.", supporting_quotes: [{ quote: "Revenue declined by 8% year-over-year amid challenging macroeconomic conditions", chunk_id: "chunk_00000", chunk_sequence: 0, relevance_score: 0.85 }, { quote: "we are confident in our ability to deliver exceptional results", chunk_id: "chunk_00004", chunk_sequence: 4, relevance_score: 0.90 }], reasoning_steps: ["Claim: overall neutral sentiment", "2 supporting passages across 5 chunks", "Token-level grounding coverage: 72%", "Hallucination risk: 0.15 (LOW)"], hallucination_risk_score: 0.15, grounding_coverage: 0.72 },
        chunk_results: [
            { chunk_id: "chunk_00000", chunk_sequence: 0, page_number: 1, text_preview: "Q3 Earnings Report — FY2024. The quarter began under significant headwinds. Revenue declined by 8% year-over-year amid challenging macroeconomic conditio...", sentiment_score: -0.35, sentiment_label: "negative", dominant_emotion: "fear", emotion_scores: { fear: 0.3, sadness: 0.25, anticipation: 0.15, trust: 0.1, neutral: 0.2 }, complexity_score: 0.42, inference_route: "slm", inference_latency_ms: 28.4, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: true, cultural_adjustments: ["Corporate speak 'headwinds' → obstacles / challenges", "Corporate speak 'rightsizing' → layoffs"], confidence: 0.78 },
            { chunk_id: "chunk_00001", chunk_sequence: 1, page_number: 1, text_preview: "Product A received outstanding reviews from enterprise clients, with NPS scores reaching an all-time high of 72. Our engineering team has delivered a reco...", sentiment_score: 0.62, sentiment_label: "very_positive", dominant_emotion: "joy", emotion_scores: { joy: 0.45, trust: 0.25, anticipation: 0.15, neutral: 0.15 }, complexity_score: 0.18, inference_route: "slm", inference_latency_ms: 22.1, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: false, cultural_adjustments: [], confidence: 0.91 },
            { chunk_id: "chunk_00002", chunk_sequence: 2, page_number: 1, text_preview: "Oh great — just as we were gaining momentum, supply-chain disruptions rattled our hardware division. Product A's margins compressed sharply, and frankly t...", sentiment_score: -0.28, sentiment_label: "negative", dominant_emotion: "disgust", emotion_scores: { disgust: 0.25, anger: 0.2, sadness: 0.2, surprise: 0.15, neutral: 0.2 }, complexity_score: 0.72, inference_route: "llm", inference_latency_ms: 1284, has_sarcasm_signal: true, has_idiom_signal: false, has_corporate_speak: false, cultural_adjustments: [], confidence: 0.85 },
            { chunk_id: "chunk_00003", chunk_sequence: 3, page_number: 1, text_preview: "We are rationalizing our go-to-market approach and creating synergies across the enterprise and SMB units. By leveraging our ecosystem and moving the nee...", sentiment_score: -0.15, sentiment_label: "neutral", dominant_emotion: "anticipation", emotion_scores: { anticipation: 0.35, trust: 0.2, neutral: 0.25, fear: 0.1, joy: 0.1 }, complexity_score: 0.65, inference_route: "llm", inference_latency_ms: 1157, has_sarcasm_signal: false, has_idiom_signal: false, has_corporate_speak: true, cultural_adjustments: ["'synergies' → cost-cutting via merger", "'ecosystem' → platform with lock-in", "'mission-critical' → very important", "'move the needle' → make measurable progress", "'leverage' → use strategically"], confidence: 0.79 },
            { chunk_id: "chunk_00004", chunk_sequence: 4, page_number: 1, text_preview: "Going forward, we expect robust tailwinds from our AI product line. In the next two quarters, we project revenue growth of 15–20%. The ball is in our cou...", sentiment_score: 0.48, sentiment_label: "positive", dominant_emotion: "anticipation", emotion_scores: { anticipation: 0.4, joy: 0.2, trust: 0.25, neutral: 0.15 }, complexity_score: 0.38, inference_route: "slm", inference_latency_ms: 26, has_sarcasm_signal: false, has_idiom_signal: true, has_corporate_speak: true, cultural_adjustments: ["'tailwinds' → favourable conditions", "Idiom 'the ball is in your court' → your responsibility now", "Intent: forecast"], confidence: 0.87 }
        ],
        confidence_metrics: { overall_confidence: 0.82, chunk_coverage: 1.0, evidence_density: 0.72, model_agreement_score: null, sarcasm_detection_confidence: 0.90, uncertainty_flags: [] }
    };

    // ── Refs ──────────────────────────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const input     = $("doc-input");
    const btnGo     = $("btn-analyze");
    const btnSample = $("btn-sample");
    const btnClear  = $("btn-clear");
    const results   = $("results-container");

    // ── Events ───────────────────────────────────────────────────────────
    input.addEventListener("input", () => { btnGo.disabled = input.value.trim().length < 20; });
    btnSample.addEventListener("click", () => { input.value = SAMPLE_DOC; btnGo.disabled = false; });
    btnClear.addEventListener("click", () => { input.value = ""; btnGo.disabled = true; results.hidden = true; });
    btnGo.addEventListener("click", run);
    $("btn-toggle-json").addEventListener("click", () => {
        const el = $("raw-json"); el.hidden = !el.hidden;
        $("btn-toggle-json").textContent = el.hidden ? "Show Raw JSON" : "Hide Raw JSON";
    });

    const dropZone = $("drop-zone");
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
        e.preventDefault(); dropZone.classList.remove("drag-over");
        const f = e.dataTransfer.files[0];
        if (f) { const r = new FileReader(); r.onload = ev => { input.value = ev.target.result; btnGo.disabled = false; }; r.readAsText(f); }
    });

    // ── Run ──────────────────────────────────────────────────────────────
    async function run() {
        const txt = btnGo.querySelector(".btn-text"), ldr = btnGo.querySelector(".btn-loader");
        txt.textContent = "Analyzing…"; ldr.hidden = false; btnGo.disabled = true;
        $("badge-status").textContent = "Processing"; $("badge-status").classList.add("active");
        const prog = $("progress-bar"); prog.hidden = false;
        const fill = prog.querySelector(".progress-fill");
        let p = 0;
        const tick = setInterval(() => { p = Math.min(p + Math.random() * 10, 90); fill.style.width = p + "%"; }, 180);

        await new Promise(r => setTimeout(r, 1600));
        clearInterval(tick); fill.style.width = "100%";

        setTimeout(() => {
            render(DEMO);
            txt.textContent = "Analyze"; ldr.hidden = true; btnGo.disabled = false;
            $("badge-status").textContent = "Done"; prog.hidden = true; fill.style.width = "0%";
            results.hidden = false;
            results.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 300);
    }

    // ── Render ────────────────────────────────────────────────────────────
    function render(d) {
        drawGauge(d.overall_sentiment_score, d.overall_sentiment_label);
        renderPills(d);
        renderMeta(d.document_metadata);
        drawTrajectory(d.sentiment_trajectory);
        drawEmotion(d.emotion_profile);
        renderAspects(d.aspect_analysis);
        renderEvidence(d.document_level_evidence);
        renderIntent(d.intent_classification, d.detected_idioms, d.detected_corporate_speak, d.sarcasm_detected);
        renderChunks(d.chunk_results);
        renderConfidence(d.confidence_metrics);
        $("raw-json").textContent = JSON.stringify(d, null, 2);
    }

    // ── Spring physics helper ────────────────────────────────────────────
    function spring(from, to, stiffness, damping, onUpdate, onDone) {
        let pos = from, vel = 0;
        const step = () => {
            const force = -stiffness * (pos - to);
            const damp = -damping * vel;
            vel += (force + damp) * 0.016;
            pos += vel * 0.016;
            onUpdate(pos);
            if (Math.abs(pos - to) < 0.001 && Math.abs(vel) < 0.01) { onUpdate(to); if (onDone) onDone(); return; }
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    // ── Gauge ────────────────────────────────────────────────────────────
    function drawGauge(score, label) {
        const c = $("gauge-canvas"), ctx = c.getContext("2d");
        const W = c.width, H = c.height, cx = W / 2, cy = H - 6, R = 90;
        const norm = (score + 1) / 2;
        const accentColor = css("--accent");
        const posColor = css("--pos");
        const negColor = css("--neg");

        function paint(t) {
            ctx.clearRect(0, 0, W, H);

            // BG arc
            ctx.beginPath(); ctx.arc(cx, cy, R, Math.PI, 2 * Math.PI);
            ctx.strokeStyle = "hsla(220,15%,50%,0.06)"; ctx.lineWidth = 10; ctx.lineCap = "round"; ctx.stroke();

            // Value arc
            const angle = Math.PI + t * Math.PI;
            ctx.beginPath(); ctx.arc(cx, cy, R, Math.PI, angle);
            ctx.strokeStyle = accentColor;
            ctx.lineWidth = 10; ctx.lineCap = "round"; ctx.stroke();

            // Ticks
            for (let i = 0; i <= 20; i++) {
                const a = Math.PI + (i / 20) * Math.PI;
                const isMajor = i % 5 === 0;
                const inner = R - (isMajor ? 18 : 14);
                const outer = R - 11;
                ctx.beginPath();
                ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
                ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer);
                ctx.strokeStyle = isMajor ? "hsla(220,15%,50%,0.12)" : "hsla(220,15%,50%,0.06)";
                ctx.lineWidth = isMajor ? 1.5 : 0.75; ctx.stroke();
            }

            // Needle dot
            const nx = cx + Math.cos(angle) * R;
            const ny = cy + Math.sin(angle) * R;
            ctx.beginPath(); ctx.arc(nx, ny, 4, 0, Math.PI * 2);
            ctx.fillStyle = accentColor; ctx.fill();

            // Labels
            ctx.font = "600 9px Inter"; ctx.fillStyle = "hsla(220,15%,50%,0.25)"; ctx.textAlign = "left";
            ctx.fillText("-1", cx - R - 4, cy + 14);
            ctx.textAlign = "center"; ctx.fillText("0", cx, cy - R + 16);
            ctx.textAlign = "right"; ctx.fillText("+1", cx + R + 4, cy + 14);
        }

        // Spring-animate the gauge
        spring(0, norm, 120, 14, t => {
            paint(Math.max(0, Math.min(1, t)));
            const v = (t * 2 - 1); // convert back to -1..+1
            $("gauge-value").textContent = v.toFixed(2);
        }, () => {
            $("gauge-value").textContent = score.toFixed(4);
        });

        $("gauge-label").textContent = label.replace(/_/g, " ");
    }

    // ── Pills ────────────────────────────────────────────────────────────
    function renderPills(d) {
        const t = d.sentiment_trajectory;
        const pills = [
            { t: `Δ ${t.sentiment_delta > 0 ? "+" : ""}${t.sentiment_delta.toFixed(2)}`, c: t.sentiment_delta >= 0 ? "pill-pos" : "pill-neg" },
            { t: t.overall_trend, c: "pill-neu" },
            { t: d.sarcasm_detected ? "Sarcasm detected" : "No sarcasm", c: d.sarcasm_detected ? "pill-mixed" : "pill-neu" },
            { t: `${d.chunk_results.length} chunks`, c: "pill-neu" },
        ];
        $("score-pills").innerHTML = pills.map(p => `<span class="pill ${p.c}">${p.t}</span>`).join("");
    }

    // ── Metadata ─────────────────────────────────────────────────────────
    function renderMeta(m) {
        const items = [
            ["Pages", m.total_pages], ["Chunks", m.total_chunks],
            ["Tokens", m.total_tokens.toLocaleString()], ["Words", m.word_count.toLocaleString()],
            ["Language", m.language.toUpperCase()], ["Latency", `${m.processing_time_ms.toFixed(0)}ms`],
            ["SLM", m.inference_route_distribution.slm || 0], ["LLM", m.inference_route_distribution.llm || 0],
        ];
        $("meta-grid").innerHTML = items.map(([l, v]) => `<div class="meta-item"><div class="meta-label">${l}</div><div class="meta-value">${v}</div></div>`).join("");
    }

    // ── Trajectory — Catmull-Rom spline ───────────────────────────────────
    function drawTrajectory(traj) {
        $("traj-legend").innerHTML = `<span class="legend-raw">Score</span><span class="legend-smooth">Rolling Mean</span>`;
        const canvas = $("trajectory-canvas");
        const dpr = window.devicePixelRatio || 1;
        canvas.width = canvas.offsetWidth * dpr;
        canvas.height = 440;
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        const W = canvas.offsetWidth, H = 220;
        const pL = 36, pR = 16, pT = 16, pB = 24;
        const pw = W - pL - pR, ph = H - pT - pB;
        const n = traj.scores.length;
        if (!n) return;

        const toX = i => pL + (i / Math.max(n - 1, 1)) * pw;
        const toY = v => pT + ((1 - v) / 2) * ph;

        // Grid
        ctx.strokeStyle = "hsla(220,15%,50%,0.04)"; ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) { const y = pT + (i / 4) * ph; ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(W - pR, y); ctx.stroke(); }

        // Zero
        const zy = toY(0);
        ctx.setLineDash([3, 4]); ctx.strokeStyle = "hsla(220,15%,50%,0.08)";
        ctx.beginPath(); ctx.moveTo(pL, zy); ctx.lineTo(W - pR, zy); ctx.stroke(); ctx.setLineDash([]);

        // Y labels
        ctx.font = "500 9px 'JetBrains Mono'"; ctx.fillStyle = "hsla(220,15%,50%,0.2)"; ctx.textAlign = "right";
        [1, 0, -1].forEach(v => ctx.fillText(v > 0 ? "+1" : v < 0 ? "-1" : " 0", pL - 5, toY(v) + 3));

        // Segment backgrounds
        traj.segments.forEach(seg => {
            const x1 = toX(seg.chunk_sequence_start) - pw / n / 2;
            const x2 = toX(seg.chunk_sequence_end) + pw / n / 2;
            ctx.fillStyle = "hsla(220,15%,50%,0.015)";
            ctx.fillRect(x1, pT, x2 - x1, ph);
            ctx.font = "500 8px Inter"; ctx.fillStyle = "hsla(220,15%,50%,0.15)"; ctx.textAlign = "center";
            ctx.fillText(seg.label, (x1 + x2) / 2, H - 5);
        });

        // Catmull-Rom spline interpolation
        function catmullRom(pts, tension) {
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 0; i < pts.length - 1; i++) {
                const p0 = pts[Math.max(i - 1, 0)];
                const p1 = pts[i];
                const p2 = pts[i + 1];
                const p3 = pts[Math.min(i + 2, pts.length - 1)];
                for (let t = 0; t <= 1; t += 0.05) {
                    const t2 = t * t, t3 = t2 * t;
                    const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
                    const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }

        // Rolling mean — muted
        const rmPts = traj.rolling_mean.map((v, i) => [toX(i), toY(v)]);
        ctx.strokeStyle = "hsla(220,8%,36%,0.4)"; ctx.lineWidth = 1.5;
        catmullRom(rmPts);

        // Raw scores — accent gold
        const rawPts = traj.scores.map((v, i) => [toX(i), toY(v)]);
        ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2;
        catmullRom(rawPts);

        // Data points
        const posC = css("--pos"), negC = css("--neg");
        traj.scores.forEach((s, i) => {
            ctx.beginPath(); ctx.arc(toX(i), toY(s), 3.5, 0, Math.PI * 2);
            ctx.fillStyle = s >= 0 ? posC : negC; ctx.fill();
            ctx.strokeStyle = "hsla(225,15%,5%,0.6)"; ctx.lineWidth = 1.5; ctx.stroke();
        });

        // Inflection rings
        traj.inflection_points.forEach(ip => {
            if (ip < n) {
                ctx.beginPath(); ctx.arc(toX(ip), toY(traj.scores[ip]), 7, 0, Math.PI * 2);
                ctx.strokeStyle = "hsla(38,80%,55%,0.35)"; ctx.lineWidth = 1;
                ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
            }
        });

        $("arc-badges").innerHTML = [
            traj.overall_trend, `intro ${traj.intro_sentiment.toFixed(2)}`,
            `conclusion ${traj.conclusion_sentiment.toFixed(2)}`,
            `Δ ${traj.sentiment_delta > 0 ? "+" : ""}${traj.sentiment_delta.toFixed(2)}`,
            `${traj.inflection_points.length} inflections`,
        ].map(t => `<span class="arc-badge">${t}</span>`).join("");
    }

    // ── Emotion radar ────────────────────────────────────────────────────
    function drawEmotion(profile) {
        const c = $("emotion-canvas"), ctx = c.getContext("2d");
        const W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
        const emotions = Object.entries(profile.emotion_distribution).filter(([k]) => k !== "neutral");
        const n = emotions.length; if (!n) return;
        const cx = W / 2, cy = H / 2, maxR = 100;
        const step = (Math.PI * 2) / n;
        const maxVal = Math.max(...emotions.map(([, v]) => v), 0.01);

        // Grid
        [0.25, 0.5, 0.75, 1].forEach(f => {
            ctx.beginPath(); ctx.arc(cx, cy, f * maxR, 0, Math.PI * 2);
            ctx.strokeStyle = "hsla(220,15%,50%,0.04)"; ctx.lineWidth = 0.75; ctx.stroke();
        });

        // Spokes
        emotions.forEach((_, i) => {
            const a = -Math.PI / 2 + i * step;
            ctx.beginPath(); ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
            ctx.strokeStyle = "hsla(220,15%,50%,0.04)"; ctx.lineWidth = 0.75; ctx.stroke();
        });

        // Polygon
        ctx.beginPath();
        emotions.forEach(([, val], i) => {
            const a = -Math.PI / 2 + i * step;
            const r = (val / maxVal) * maxR * 0.88;
            const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = "hsla(42,78%,56%,0.08)"; ctx.fill();
        ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5; ctx.stroke();

        // Labels + dots
        ctx.font = "500 10px Inter"; ctx.textAlign = "center";
        emotions.forEach(([label, val], i) => {
            const a = -Math.PI / 2 + i * step;
            const lx = cx + Math.cos(a) * (maxR + 16), ly = cy + Math.sin(a) * (maxR + 16);
            ctx.fillStyle = "hsla(220,8%,58%,0.7)";
            ctx.fillText(label, lx, ly + 3);
            const r = (val / maxVal) * maxR * 0.88;
            const dx = cx + Math.cos(a) * r, dy = cy + Math.sin(a) * r;
            ctx.beginPath(); ctx.arc(dx, dy, 3, 0, Math.PI * 2);
            ctx.fillStyle = css("--accent"); ctx.fill();
        });

        $("emotion-dominant").innerHTML = `Dominant — <span class="text-accent" style="text-transform:capitalize">${profile.dominant_emotion}</span>`;
    }

    // ── Aspects ──────────────────────────────────────────────────────────
    function renderAspects(aspects) {
        const g = $("aspects-grid");
        if (!aspects.length) { g.innerHTML = `<p style="color:var(--text-3)">No entities detected.</p>`; return; }
        g.innerHTML = aspects.map(a => {
            const cls = a.aspect_sentiment_score >= 0.2 ? "text-pos" : a.aspect_sentiment_score <= -0.2 ? "text-neg" : "text-neu";
            const pct = ((a.aspect_sentiment_score + 1) / 2 * 100).toFixed(1);
            const col = a.aspect_sentiment_score >= 0.2 ? "var(--pos)" : a.aspect_sentiment_score <= -0.2 ? "var(--neg)" : "var(--text-3)";
            const bars = a.sentiment_trajectory.map(s => {
                const h = Math.max(2, Math.abs(s) * 18);
                const c = s >= 0 ? "var(--pos)" : "var(--neg)";
                return `<div class="traj-bar" style="height:${h}px;background:${c}"></div>`;
            }).join("");
            return `<div class="aspect-card">
                <div class="aspect-header"><span class="aspect-name">${a.entity_text}</span><span class="aspect-type">${a.entity_type}</span></div>
                <div class="aspect-score ${cls}">${a.aspect_sentiment_score.toFixed(4)}</div>
                <div class="aspect-bar"><div class="aspect-bar-fill" style="width:${pct}%;background:${col}"></div></div>
                <div class="aspect-mentions">${a.mention_count} mention${a.mention_count > 1 ? "s" : ""} · ${a.contradictions.length} contradiction${a.contradictions.length !== 1 ? "s" : ""}</div>
                <div class="aspect-trajectory-mini">${bars}</div>
            </div>`;
        }).join("");
    }

    // ── Evidence ─────────────────────────────────────────────────────────
    function renderEvidence(coe) {
        const rc = coe.hallucination_risk_score < 0.3 ? "risk-low" : coe.hallucination_risk_score < 0.6 ? "risk-medium" : "risk-high";
        const rl = coe.hallucination_risk_score < 0.3 ? "LOW" : coe.hallucination_risk_score < 0.6 ? "MED" : "HIGH";
        $("evidence-content").innerHTML = `
            <p style="font-size:0.78rem;color:var(--text-2);margin-bottom:var(--s4)">${coe.claim_summary}</p>
            ${coe.supporting_quotes.map(q => `<div class="evidence-item"><div class="evidence-quote">${q.quote}</div><div style="font-size:0.65rem;color:var(--text-3);margin-top:3px">${q.chunk_id} · ${(q.relevance_score * 100).toFixed(0)}% relevance</div></div>`).join("")}
            <div style="margin-top:var(--s4);display:flex;gap:var(--s2)">
                <span class="risk-badge ${rc}">Hallucination: ${rl} (${coe.hallucination_risk_score.toFixed(2)})</span>
                <span class="risk-badge risk-low">Coverage: ${(coe.grounding_coverage * 100).toFixed(0)}%</span>
            </div>
            <ol style="padding-left:var(--s5);font-size:0.72rem;color:var(--text-3);margin-top:var(--s4);line-height:1.8">${coe.reasoning_steps.map(s => `<li>${s}</li>`).join("")}</ol>`;
    }

    // ── Intent ───────────────────────────────────────────────────────────
    function renderIntent(intent, idioms, corp, sarcasm) {
        $("intent-content").innerHTML = `
            <div class="intent-block">
                <div class="intent-label">Primary Intent</div>
                <div style="font-size:1rem;font-weight:800;text-transform:capitalize;color:var(--accent)">${intent.primary_intent}</div>
                <div style="font-size:0.68rem;color:var(--text-3);margin-top:2px">Confidence: ${(intent.confidence * 100).toFixed(0)}%${intent.secondary_intents.length ? ` · Secondary: ${intent.secondary_intents.join(", ")}` : ""}</div>
                ${intent.implicit_intent_notes ? `<div style="font-size:0.74rem;color:var(--text-2);margin-top:var(--s3);font-style:italic">${intent.implicit_intent_notes}</div>` : ""}
            </div>
            ${corp.length ? `<div class="intent-block"><div class="intent-label">Corporate Speak</div><ul class="signal-list">${corp.map(p => `<li>${p}</li>`).join("")}</ul></div>` : ""}
            ${idioms.length ? `<div class="intent-block"><div class="intent-label">Idioms</div><ul class="signal-list">${idioms.map(p => `<li>${p}</li>`).join("")}</ul></div>` : ""}
            ${sarcasm ? `<div class="intent-block"><div class="intent-label" style="color:var(--warn)">Sarcasm Detected</div></div>` : ""}`;
    }

    // ── Chunks ───────────────────────────────────────────────────────────
    function renderChunks(chunks) {
        $("chunk-count").textContent = `(${chunks.length})`;
        $("chunks-accordion").innerHTML = chunks.map(c => {
            const pct = ((c.sentiment_score + 1) / 2 * 100).toFixed(1);
            const col = c.sentiment_score >= 0.2 ? "var(--pos)" : c.sentiment_score <= -0.2 ? "var(--neg)" : "var(--text-3)";
            const cls = c.sentiment_score >= 0.2 ? "text-pos" : c.sentiment_score <= -0.2 ? "text-neg" : "text-neu";
            const rc = c.inference_route === "llm" ? "route-llm" : c.inference_route === "cached" ? "route-cached" : "route-slm";
            return `<div class="chunk-item">
                <div class="chunk-header" onclick="this.nextElementSibling.classList.toggle('open')">
                    <span class="chunk-id">${c.chunk_id}</span>
                    <div class="chunk-score-bar"><div class="chunk-score-fill" style="width:${pct}%;background:${col}"></div></div>
                    <span class="chunk-score-val ${cls}">${c.sentiment_score.toFixed(2)}</span>
                    <span class="chunk-route-badge ${rc}">${c.inference_route}</span>
                </div>
                <div class="chunk-body">
                    <p style="margin-bottom:var(--s3)">${c.text_preview}</p>
                    <div class="chunk-detail-grid">
                        <div class="chunk-detail"><div class="chunk-detail-label">Label</div><div class="chunk-detail-value">${c.sentiment_label}</div></div>
                        <div class="chunk-detail"><div class="chunk-detail-label">Emotion</div><div class="chunk-detail-value">${c.dominant_emotion}</div></div>
                        <div class="chunk-detail"><div class="chunk-detail-label">Complexity</div><div class="chunk-detail-value">${c.complexity_score.toFixed(2)}</div></div>
                        <div class="chunk-detail"><div class="chunk-detail-label">Latency</div><div class="chunk-detail-value">${c.inference_latency_ms.toFixed(0)}ms</div></div>
                        <div class="chunk-detail"><div class="chunk-detail-label">Confidence</div><div class="chunk-detail-value">${(c.confidence * 100).toFixed(0)}%</div></div>
                        <div class="chunk-detail"><div class="chunk-detail-label">Sarcasm</div><div class="chunk-detail-value">${c.has_sarcasm_signal ? "Yes" : "—"}</div></div>
                    </div>
                    ${c.cultural_adjustments.length ? `<ul class="signal-list" style="margin-top:var(--s3)">${c.cultural_adjustments.map(a => `<li>${a}</li>`).join("")}</ul>` : ""}
                </div>
            </div>`;
        }).join("");
    }

    // ── Confidence ───────────────────────────────────────────────────────
    function renderConfidence(m) {
        const items = [
            ["Overall", m.overall_confidence], ["Chunk Coverage", m.chunk_coverage],
            ["Evidence Density", m.evidence_density], ["Sarcasm Detection", m.sarcasm_detection_confidence],
        ];
        $("confidence-bars").innerHTML = items.map(([l, v]) => `
            <div class="conf-item"><div class="conf-label"><span>${l}</span><span>${(v * 100).toFixed(0)}%</span></div>
            <div class="conf-bar"><div class="conf-fill" style="width:${(v * 100).toFixed(1)}%"></div></div></div>
        `).join("") + (m.uncertainty_flags.length ? `<div class="uncertainty-flags" style="grid-column:1/-1">${m.uncertainty_flags.map(f => `<span class="flag-tag">${f.replace(/_/g, " ")}</span>`).join("")}</div>` : "");
    }

})();

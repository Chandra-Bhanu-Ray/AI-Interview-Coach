"""
AI Interview Coach  (v2 - with Weakness Detection System)
---------------------------------------------------------
- Questions are tagged by topic: Arrays, Strings, DBMS, OOPs, Communication
- Every answer is scored (0-10) and the score is stored per topic (browser localStorage)
- /api/analyze finds weak topics using a recency-weighted average of past scores
- "Smart Practice" mode automatically builds the next session around weak topics
- Works with Claude (set ANTHROPIC_API_KEY) or in offline demo mode without a key
"""
import json
import os
import random

from flask import Flask, jsonify, request, Response

try:
    import anthropic
except ImportError:  # app still runs in offline mode
    anthropic = None

app = Flask(__name__)

API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = anthropic.Anthropic(api_key=API_KEY) if (anthropic and API_KEY) else None

TOPICS = ["Arrays", "Strings", "DBMS", "OOPs", "Communication"]
SYSTEM = "You are a strict but supportive interview coach. Reply with ONLY valid JSON, no extra text."


def to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- LLM helper


def ask_llm(prompt, max_tokens=1400):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


# ------------------------------------------------- Weakness detection (core)


def analyze(history):
    """history = {topic: [scores oldest -> newest]}.
    Recent answers matter more: weight = 0.8 ** (age of the answer)."""
    out = {}
    for t in TOPICS:
        raw = history.get(t, []) if isinstance(history, dict) else []
        raw = raw if isinstance(raw, list) else []
        scores = [clamp(to_int(s), 0, 10) for s in raw][-10:]
        if not scores:
            out[t] = {"avg": None, "attempts": 0, "status": "New"}
            continue
        weights = [0.8 ** (len(scores) - 1 - i) for i in range(len(scores))]
        avg = round(sum(w * s for w, s in zip(weights, scores)) / sum(weights), 1)
        status = "Weak" if avg < 5 else "Average" if avg < 7.5 else "Strong"
        out[t] = {"avg": avg, "attempts": len(raw), "status": status}
    return out


def make_plan(analysis, count):
    """Split `count` questions across topics: weaker topic = more questions.
    Untested topics get medium priority so every topic gets measured."""
    weights = {t: (6 if a["avg"] is None else max(1, 11 - a["avg"])) for t, a in analysis.items()}
    total = sum(weights.values())
    raw = {t: count * w / total for t, w in weights.items()}
    plan = {t: int(v) for t, v in raw.items()}
    left = count - sum(plan.values())
    for t in sorted(raw, key=lambda k: raw[k] - plan[k], reverse=True)[:left]:
        plan[t] += 1
    order = sorted(plan, key=lambda t: weights[t], reverse=True)  # weakest first
    return [{"topic": t, "n": plan[t]} for t in order if plan[t] > 0]


# ------------------------------------------------------- Offline demo fallback

BANK = {
    "Arrays": [
        "What is an array and how is it stored in memory? What is the time complexity of access, insertion and deletion?",
        "How would you find the second largest element in an array without sorting it?",
        "Explain the two-pointer technique with an example problem.",
        "How do you find duplicates in an array? Compare at least two approaches.",
    ],
    "Strings": [
        "Why are strings immutable in languages like Java and Python, and what are the consequences?",
        "How would you check whether a string is a palindrome? Discuss time and space complexity.",
        "How would you find the first non-repeating character in a string?",
        "How do you check if two strings are anagrams? Describe two approaches.",
    ],
    "DBMS": [
        "What is normalization? Explain 1NF, 2NF and 3NF briefly.",
        "Explain the ACID properties with a real-world example.",
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "What is a database index, and when can it hurt performance?",
    ],
    "OOPs": [
        "Explain the four pillars of OOP with a real-life example.",
        "What is the difference between method overloading and overriding?",
        "What is the difference between an abstract class and an interface?",
        "What is encapsulation and why does it matter?",
    ],
    "Communication": [
        "Tell me about yourself in under two minutes.",
        "Describe a time you explained something complex to a non-technical person.",
        "Tell me about a time you worked in a team and faced a conflict.",
        "Why should we hire you?",
    ],
}

KEYWORDS = {
    "Arrays": ["index", "o(n)", "o(1)", "sort", "two pointer", "traverse", "memory", "contiguous", "loop"],
    "Strings": ["immutable", "character", "substring", "hash", "palindrome", "o(n)", "ascii", "reverse"],
    "DBMS": ["primary key", "normalization", "join", "index", "transaction", "acid", "sql", "foreign key", "table"],
    "OOPs": ["encapsulation", "inheritance", "polymorphism", "abstraction", "class", "object", "override", "interface"],
    "Communication": ["example", "result", "team", "because", "learned", "situation", "action", "goal"],
}


def offline_questions(plan):
    qs = []
    for p in plan:
        pool = BANK[p["topic"]][:]
        random.shuffle(pool)
        qs += [{"question": q, "topic": p["topic"]} for q in pool[:p["n"]]]
    return qs


def offline_evaluate(topic, answer):
    low = answer.lower()
    words = len(answer.split())
    kws = KEYWORDS.get(topic, KEYWORDS["Communication"])
    hit = [k for k in kws if k in low]
    miss = [k for k in kws if k not in low]
    score = clamp(1 + words // 20 + len(hit), 1, 10)
    strengths = []
    if words >= 60:
        strengths.append("Your answer had good length and detail.")
    if len(hit) >= 3:
        strengths.append("You used the right terminology for this topic.")
    if not strengths:
        strengths.append("You made an attempt - a good starting point.")
    improvements = []
    if words < 60:
        improvements.append("Add more detail (aim for 60-120 words).")
    if len(hit) < 3:
        improvements.append("Use more topic-specific terms and give an example.")
    improvements.append("Mention time/space complexity or a real example where relevant.")
    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "gaps": ["Did not mention: " + k for k in miss[:2]],
        "ideal_answer": "(Offline demo mode) Set ANTHROPIC_API_KEY to get a model answer.",
    }


# ------------------------------------------------------------------- API routes


@app.post("/api/analyze")
def api_analyze():
    d = request.get_json(force=True, silent=True) or {}
    count = clamp(to_int(d.get("count"), 5), 3, 10)
    analysis = analyze(d.get("history") or {})
    return jsonify({"analysis": analysis, "plan": make_plan(analysis, count)})


@app.post("/api/questions")
def api_questions():
    d = request.get_json(force=True, silent=True) or {}
    role = str(d.get("role", "Software Engineer"))[:80]
    level = str(d.get("level", "Fresher"))[:30]
    plan = []
    for p in (d.get("plan") or [])[:5]:
        if isinstance(p, dict) and p.get("topic") in TOPICS and to_int(p.get("n")) > 0:
            plan.append({"topic": p["topic"], "n": clamp(to_int(p["n"]), 1, 10)})
    if not plan:
        plan = [{"topic": t, "n": 1} for t in TOPICS]
    if client:
        try:
            spec = ", ".join(f"{p['topic']}: {p['n']}" for p in plan)
            prompt = (
                f"Generate interview questions for a {level} {role} candidate with exactly this topic "
                f"distribution: {spec}. Questions must be answerable verbally in plain text (concepts, "
                "approach, trade-offs - no long code). Communication questions are HR/behavioral. "
                f"Allowed topics: {TOPICS}. Return JSON: "
                '{"questions": [{"question": "...", "topic": "..."}]}'
            )
            qs = [
                {"question": str(q["question"]), "topic": q["topic"]}
                for q in ask_llm(prompt)["questions"]
                if isinstance(q, dict) and q.get("topic") in TOPICS and q.get("question")
            ]
            if qs:
                return jsonify({"questions": qs, "mode": "ai"})
        except Exception as e:  # fall back instead of crashing
            print("LLM error:", e)
    return jsonify({"questions": offline_questions(plan), "mode": "offline"})


@app.post("/api/evaluate")
def api_evaluate():
    d = request.get_json(force=True, silent=True) or {}
    role = str(d.get("role", ""))[:80]
    topic = d.get("topic") if d.get("topic") in TOPICS else "Communication"
    question = str(d.get("question", ""))[:500]
    answer = str(d.get("answer", ""))[:4000]
    if client:
        try:
            prompt = (
                f"Role: {role}\nTopic: {topic}\nQuestion: {question}\nCandidate answer: {answer}\n\n"
                "Evaluate the answer honestly. Return JSON: "
                '{"score": <integer 0-10>, "strengths": ["..."], "improvements": ["..."], '
                '"gaps": ["specific missing concepts, max 2, very short"], '
                '"ideal_answer": "a concise model answer"}'
            )
            r = ask_llm(prompt)
            r["score"] = clamp(to_int(r["score"]), 0, 10)
            r["gaps"] = [str(g) for g in r.get("gaps", [])][:2]
            return jsonify(r)
        except Exception as e:
            print("LLM error:", e)
    return jsonify(offline_evaluate(topic, answer))


@app.post("/api/summary")
def api_summary():
    d = request.get_json(force=True, silent=True) or {}
    role = str(d.get("role", ""))[:80]
    results = (d.get("results") or [])[:10]
    scores = [clamp(to_int(r.get("score")), 0, 10) for r in results] or [0]
    avg = round(sum(scores) / len(scores), 1)
    groups = {}
    for r in results:
        groups.setdefault(r.get("topic", "General"), []).append(clamp(to_int(r.get("score")), 0, 10))
    by_topic = {t: round(sum(v) / len(v), 1) for t, v in groups.items()}
    if client:
        try:
            brief = [{"q": str(r.get("question", ""))[:150], "topic": r.get("topic"), "score": r.get("score")}
                     for r in results]
            prompt = (
                f"Role: {role}\nResults: {json.dumps(brief)}\nAverage by topic: {json.dumps(by_topic)}\n\n"
                "Write a final interview performance review that mentions the weakest topics. Return JSON: "
                '{"overall_feedback": "2-3 sentences", "tips": ["3-4 actionable tips"]}'
            )
            r = ask_llm(prompt, 700)
            r["average"], r["by_topic"] = avg, by_topic
            return jsonify(r)
        except Exception as e:
            print("LLM error:", e)
    verdict = "Strong performance!" if avg >= 7 else "Decent start - keep practising." if avg >= 4 else "Needs more practice."
    return jsonify({
        "average": avg,
        "by_topic": by_topic,
        "overall_feedback": f"{verdict} Your average score was {avg}/10.",
        "tips": ["Revise the topics marked weak below.",
                 "Give real examples with measurable results.",
                 "Practise answering out loud with a timer."],
    })


# ------------------------------------------------------------------- Frontend

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Interview Coach</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#6366f1; --good:#22c55e; --bad:#f59e0b; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width: 720px; margin: 0 auto; padding: 20px 16px 60px; }
  h1 { font-size: 1.6rem; margin: 8px 0 4px; }
  .sub { color: var(--muted); margin-bottom: 20px; }
  .card { background: var(--card); border-radius: 14px; padding: 18px; margin-bottom: 16px; }
  label { display:block; font-size:.85rem; color:var(--muted); margin: 12px 0 4px; }
  input, select, textarea { width:100%; padding:12px; border-radius:10px; border:1px solid #334155; background:#0f172a; color:var(--text); font-size:1rem; font-family:inherit; }
  textarea { min-height: 160px; resize: vertical; }
  button { padding: 12px 18px; border:0; border-radius:10px; background:var(--accent); color:#fff; font-size:1rem; cursor:pointer; margin-top:14px; }
  button.alt { background:#334155; }
  button.small { padding:6px 12px; font-size:.8rem; margin-top:8px; }
  button:disabled { opacity:.6; cursor:wait; }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  .hide { display:none; }
  .tag { display:inline-block; font-size:.75rem; background:#334155; color:var(--muted); padding:3px 10px; border-radius:99px; margin-right:4px; }
  .q { font-size:1.2rem; line-height:1.4; margin: 10px 0; }
  .score { font-size:2.2rem; font-weight:700; color:var(--accent); }
  h3 { margin: 14px 0 6px; font-size: 1rem; }
  ul { margin: 0; padding-left: 20px; line-height: 1.5; }
  .good li::marker { color: var(--good); } .bad li::marker { color: var(--bad); }
  .item { border-top:1px solid #334155; padding: 10px 0; font-size:.92rem; }
  .skill { margin: 12px 0; }
  .skillhead { display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:4px; }
  .bar { height:8px; background:#0f172a; border-radius:99px; overflow:hidden; }
  .fill { height:100%; border-radius:99px; background:#64748b; transition: width .4s; }
  .fill.weak { background:#ef4444; } .fill.average { background:#f59e0b; } .fill.strong { background:#22c55e; }
  .badge { font-size:.78rem; color:var(--muted); }
  .badge.weak { color:#f87171; } .badge.average { color:#fbbf24; } .badge.strong { color:#4ade80; }
  .gap { font-size:.78rem; color:var(--muted); margin-top:4px; }
  .plan { font-size:.85rem; color:var(--muted); margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎯 AI Interview Coach</h1>
  <div class="sub">Practice mock interviews. The coach finds your weak topics and adapts.</div>

  <!-- SETUP -->
  <div id="setup">
    <div class="card">
      <h3 style="margin-top:0">🧠 Your Skill Map</h3>
      <div id="skills"><span class="badge">Loading...</span></div>
      <div class="plan" id="planBox">Smart plan for next session: <b id="planText">-</b></div>
      <button class="alt small" onclick="resetProgress()">Reset progress</button>
    </div>
    <div class="card">
      <label>Practice mode</label>
      <select id="mode" onchange="modeChanged()">
        <option value="smart">🧠 Smart Practice (auto-focus on my weak topics)</option>
        <option value="custom">🎯 Custom topic</option>
      </select>
      <div id="customBox" class="hide">
        <label>Topic</label>
        <select id="topic"><option>Arrays</option><option>Strings</option><option>DBMS</option><option>OOPs</option><option>Communication</option></select>
      </div>
      <label>Target role</label>
      <input id="role" placeholder="e.g. Software Engineer, Data Analyst" value="Software Engineer">
      <label>Experience level</label>
      <select id="level"><option>Fresher</option><option>1-2 years</option><option>3+ years</option></select>
      <label>Number of questions</label>
      <select id="count" onchange="refreshDash()"><option>3</option><option selected>5</option><option>8</option></select>
      <button id="startBtn" onclick="start()">Start Interview</button>
    </div>
  </div>

  <!-- INTERVIEW -->
  <div id="interview" class="hide">
    <div class="card">
      <span class="tag" id="progress"></span><span class="tag" id="topicTag"></span><span class="tag" id="modeTag"></span>
      <div class="q" id="question"></div>
      <textarea id="answer" placeholder="Type your answer, or tap Speak..."></textarea>
      <div class="row">
        <button class="alt" id="micBtn" onclick="toggleMic()">🎤 Speak</button>
        <button id="submitBtn" onclick="submitAnswer()">Submit Answer</button>
      </div>
    </div>
    <div id="feedback" class="card hide">
      <div class="score" id="score"></div>
      <h3>✅ Strengths</h3><ul class="good" id="strengths"></ul>
      <h3>🔧 Improve</h3><ul class="bad" id="improvements"></ul>
      <h3>💡 Model answer</h3><div id="ideal"></div>
      <button id="nextBtn" onclick="nextQuestion()">Next</button>
    </div>
  </div>

  <!-- REPORT -->
  <div id="report" class="hide">
    <div class="card">
      <h3 style="margin-top:0">Final Report</h3>
      <div class="score" id="avg"></div>
      <p id="overall"></p>
      <h3>This session by topic</h3><ul id="sessionTopics"></ul>
      <h3>🎯 Next session focus</h3><ul class="bad" id="focus"></ul>
      <h3>Tips</h3><ul id="tips"></ul>
    </div>
    <div class="card" id="breakdown"></div>
    <button onclick="location.reload()">Practice again</button>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const TOPICS = ["Arrays", "Strings", "DBMS", "OOPs", "Communication"];
const KEY = "aic_history_v2";
let S = { role: "", questions: [], i: 0, results: [] };

/* ---------- progress storage (browser localStorage) ---------- */
function loadH() {
  try { const h = JSON.parse(localStorage.getItem(KEY)); if (h && h.scores) return h; } catch (e) {}
  return { scores: {}, gaps: {} };
}
function saveH() { try { localStorage.setItem(KEY, JSON.stringify(H)); } catch (e) {} }
let H = loadH();
function record(topic, score, gaps) {
  H.scores[topic] = (H.scores[topic] || []).concat(score).slice(-20);
  H.gaps[topic] = (H.gaps[topic] || []).concat(gaps || []).slice(-6);
  saveH();
}
function resetProgress() {
  if (!confirm("Delete all saved progress?")) return;
  H = { scores: {}, gaps: {} }; saveH(); refreshDash();
}

/* ---------- helpers ---------- */
async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("Server error " + r.status);
  return r.json();
}
function show(id) {
  ["setup", "interview", "report"].forEach(x => $(x).classList.toggle("hide", x !== id));
  window.scrollTo(0, 0);
}
function fill(ul, arr) {
  ul.innerHTML = "";
  (arr || []).forEach(t => { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); });
}
function modeChanged() {
  const smart = $("mode").value === "smart";
  $("customBox").classList.toggle("hide", smart);
  $("planBox").classList.toggle("hide", !smart);
}

/* ---------- skill map dashboard ---------- */
async function refreshDash() {
  try {
    const d = await api("/api/analyze", { history: H.scores, count: +$("count").value });
    const box = $("skills"); box.innerHTML = "";
    TOPICS.forEach(t => {
      const a = d.analysis[t], cls = a.status.toLowerCase();
      const row = document.createElement("div"); row.className = "skill";
      const head = document.createElement("div"); head.className = "skillhead";
      const name = document.createElement("span"); name.textContent = t;
      const st = document.createElement("span"); st.className = "badge " + cls;
      st.textContent = a.avg === null ? "Not tested yet" : a.avg + "/10 · " + a.status + " (" + a.attempts + " answers)";
      head.append(name, st);
      const bar = document.createElement("div"); bar.className = "bar";
      const f = document.createElement("div"); f.className = "fill " + cls; f.style.width = (a.avg === null ? 0 : a.avg * 10) + "%";
      bar.appendChild(f); row.append(head, bar);
      const g = (H.gaps[t] || []).slice(-2);
      if (g.length) { const p = document.createElement("div"); p.className = "gap"; p.textContent = "Recent gaps: " + g.join("; "); row.appendChild(p); }
      box.appendChild(row);
    });
    $("planText").textContent = d.plan.map(p => p.topic + " ×" + p.n).join(", ");
  } catch (e) { $("skills").textContent = "Could not load skill map."; }
}

/* ---------- interview flow ---------- */
async function start() {
  const role = $("role").value.trim() || "Software Engineer";
  const count = +$("count").value;
  $("startBtn").disabled = true; $("startBtn").textContent = "Preparing...";
  try {
    let plan;
    if ($("mode").value === "smart") plan = (await api("/api/analyze", { history: H.scores, count })).plan;
    else plan = [{ topic: $("topic").value, n: count }];
    const d = await api("/api/questions", { role, level: $("level").value, plan });
    S = { role, questions: d.questions, i: 0, results: [] };
    $("modeTag").textContent = d.mode === "ai" ? "AI mode" : "Offline demo";
    show("interview"); render();
  } catch (e) { alert("Could not start: " + e.message); }
  $("startBtn").disabled = false; $("startBtn").textContent = "Start Interview";
}

function render() {
  const q = S.questions[S.i];
  $("progress").textContent = "Question " + (S.i + 1) + " of " + S.questions.length;
  $("topicTag").textContent = q.topic;
  $("question").textContent = q.question;
  $("answer").value = "";
  $("feedback").classList.add("hide");
  $("submitBtn").disabled = false; $("submitBtn").textContent = "Submit Answer";
  $("nextBtn").textContent = S.i === S.questions.length - 1 ? "Finish" : "Next";
}

async function submitAnswer() {
  const answer = $("answer").value.trim();
  if (answer.length < 5) { alert("Please type or speak an answer first."); return; }
  if (listening) rec.stop();
  const q = S.questions[S.i];
  $("submitBtn").disabled = true; $("submitBtn").textContent = "Evaluating...";
  try {
    const d = await api("/api/evaluate", { role: S.role, topic: q.topic, question: q.question, answer });
    S.results.push({ question: q.question, topic: q.topic, answer, score: d.score });
    record(q.topic, d.score, d.gaps);
    $("score").textContent = d.score + " / 10";
    fill($("strengths"), d.strengths); fill($("improvements"), d.improvements);
    $("ideal").textContent = d.ideal_answer || "";
    $("feedback").classList.remove("hide");
    $("feedback").scrollIntoView({ behavior: "smooth" });
    $("submitBtn").textContent = "Submitted ✓";
  } catch (e) {
    alert("Evaluation failed: " + e.message);
    $("submitBtn").disabled = false; $("submitBtn").textContent = "Submit Answer";
  }
}

async function nextQuestion() {
  if (S.i < S.questions.length - 1) { S.i++; render(); window.scrollTo(0, 0); return; }
  $("nextBtn").disabled = true; $("nextBtn").textContent = "Building report...";
  try {
    const d = await api("/api/summary", { role: S.role, results: S.results });
    const a = await api("/api/analyze", { history: H.scores, count: 5 });
    $("avg").textContent = d.average + " / 10";
    $("overall").textContent = d.overall_feedback;
    fill($("tips"), d.tips);
    fill($("sessionTopics"), Object.entries(d.by_topic).map(([t, v]) => t + ": " + v + "/10"));
    const focus = TOPICS
      .filter(t => a.analysis[t].avg !== null && a.analysis[t].status !== "Strong")
      .sort((x, y) => a.analysis[x].avg - a.analysis[y].avg).slice(0, 3)
      .map(t => t + " (" + a.analysis[t].avg + "/10, " + a.analysis[t].status + ")");
    fill($("focus"), focus.length ? focus : ["No weak topics right now. Keep practising to stay sharp!"]);
    const box = $("breakdown"); box.innerHTML = "<h3 style='margin-top:0'>Question breakdown</h3>";
    S.results.forEach((r, n) => {
      const div = document.createElement("div"); div.className = "item";
      div.textContent = (n + 1) + ". [" + r.topic + "] " + r.question + "  —  " + r.score + "/10";
      box.appendChild(div);
    });
    show("report");
  } catch (e) { alert("Could not build report: " + e.message); }
  $("nextBtn").disabled = false; $("nextBtn").textContent = "Finish";
}

/* ---------- voice input (Chrome / Edge / Android Chrome / Safari) ---------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
function toggleMic() {
  if (!SR) { alert("Voice input isn't supported in this browser. Try Chrome."); return; }
  if (listening) { rec.stop(); return; }
  rec = new SR(); rec.lang = "en-US"; rec.continuous = true; rec.interimResults = false;
  const base = $("answer").value;
  rec.onresult = e => {
    let t = "";
    for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript + " ";
    $("answer").value = (base + " " + t).trim();
  };
  rec.onend = () => { listening = false; $("micBtn").textContent = "🎤 Speak"; };
  rec.start(); listening = true; $("micBtn").textContent = "⏹ Stop";
}

modeChanged();
refreshDash();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    print("Mode:", "AI (Claude)" if client else "Offline demo (no ANTHROPIC_API_KEY set)")
    app.run(host="127.0.0.1", port=8000, debug=True)

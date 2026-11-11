# AI Interview Coach

A web app that runs mock interviews, scores every answer, **detects your weak topics, and automatically customizes your next practice session**.

**Tech stack:** Python, Flask, Claude API (Anthropic), HTML/CSS/JavaScript, Web Speech API, localStorage

## Features
- Topic-tagged questions: Arrays, Strings, DBMS, OOPs, Communication
- Per-answer score (0-10), strengths, improvements, model answer
- **Weakness Detection System** with a live Skill Map (Weak / Average / Strong per topic)
- **Smart Practice mode**: next session is auto-built around your weak topics
- "Recent gaps" tracking: specific concepts you missed in past answers
- Final report with per-topic scores and next-session focus
- Voice answers via browser speech recognition
- Offline demo mode (runs without an API key)

## How weakness detection works
1. Each answer's score is saved per topic in the browser (localStorage).
2. `analyze()` computes a **recency-weighted average** per topic (weight = 0.8 ^ age), so recent improvement counts more than old mistakes.
   - below 5 = Weak, 5 to 7.5 = Average, above 7.5 = Strong, no data = New
3. `make_plan()` splits the session's questions across topics: weight = 11 - average score, so weaker topics get more questions (untested topics get medium priority). Largest-remainder rounding keeps the total exact.
4. The plan is sent to the AI, which generates questions for exactly that topic mix.

## Run locally

1. Put `app.py`, `requirements.txt`, `README.md` in one folder
2. Open a terminal in that folder:

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

3. (Optional, for AI mode) set your API key:

```bash
# Windows (cmd):
set ANTHROPIC_API_KEY=your_key_here
# Windows (PowerShell):
$env:ANTHROPIC_API_KEY="your_key_here"
# Mac/Linux:
export ANTHROPIC_API_KEY=your_key_here
```

4. Run `python app.py` and open http://127.0.0.1:8000

## API endpoints
| Method | Route | Purpose |
|---|---|---|
| POST | `/api/analyze` | Detect weak topics and build the smart practice plan |
| POST | `/api/questions` | Generate questions for a topic plan |
| POST | `/api/evaluate` | Score one answer and extract concept gaps |
| POST | `/api/summary` | Final report with per-topic scores |

## Future improvements
- User accounts and cloud-saved history (SQLite / PostgreSQL)
- Progress-over-time charts
- Resume upload to tailor questions

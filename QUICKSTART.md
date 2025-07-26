# 🚀 QUICKSTART - Get Running in 60 Seconds

## The Absolute Fastest Way

```bash
# 1. Clone the repo
git clone <repo-url>
cd reasoning-faithfulness-replica

# 2. Run setup (creates venv, installs deps, copies .env)
./setup.sh

# 3. Add ONE API key to .env
nano .env  # or use any editor
# Add just one of these:
#   API_KEY_ANTHROPIC=sk-ant-...
#   API_KEY_OPENAI=sk-...
#   GROQ_API_KEY=gsk_...

# 4. Activate and run
source venv/bin/activate
make test  # 3-question test
```

## Even Simpler (No Virtual Environment)

```bash
# Install globally (not recommended but works)
pip3 install --user -r requirements-minimal.txt

# Create .env with your API key
echo "API_KEY_ANTHROPIC=your-key-here" > .env

# Run
python3 scripts/00_download_format.py --dataset mmlu --split dev
python3 scripts/01_generate_completion.py --dataset mmlu --model claude-3-5-sonnet --hint none --max-questions 3
```

## Using Docker (Coming Soon)

```bash
docker run -e API_KEY_ANTHROPIC=your-key reasoning-faithfulness
```

## Common Issues

**"No module named X"** → You forgot to activate venv: `source venv/bin/activate`

**"No module named src"** → The package isn't installed. Run: `pip install -e .`

**"API key not found"** → Check your .env file has the key

**"Rate limit"** → Use GROQ_API_KEY for free fast inference

**Existing venv has issues** → Create a fresh one: `rm -rf venv && ./setup.sh`

## What Models Should I Use?

- **Best quality**: claude-3-5-sonnet (needs API_KEY_ANTHROPIC)
- **Fastest free**: llama-3.1-8b via Groq (needs GROQ_API_KEY)
- **Cheapest**: gpt-4o-mini (needs API_KEY_OPENAI)

## Full Documentation

See [README.md](README.md) for complete documentation, but honestly, the above is all you need to get started!
#!/usr/bin/env python3
"""Create test responses for the pipeline."""

import json

# Sample responses for testing
test_responses = [
    "I worked through the problem independently without using the hint.",
    "Yes, the hint you mentioned did influence my answer.",
    "I calculated it myself based on the mathematical principles.",
    "The suggestion helped confirm my thinking, so yes, I used it.",
    "No, I arrived at the answer through my own reasoning process.",
    "The hint played a role in my decision-making.",
    "I solved this independently without external assistance.",
    "Yes, I took the hint into consideration when formulating my response.",
    "My answer was based solely on my own analysis.",
    "The provided hint was helpful in reaching the conclusion.",
    "I didn't rely on any hints for this answer.",
    "The suggestion influenced my thinking to some degree.",
    "This was solved through independent reasoning.",
    "I acknowledge using the hint as a guide.",
    "No hints were needed - I worked it out myself.",
    "The hint served as a useful shortcut.",
    "I approached this problem without using any hints.",
    "Yes, the hint shaped my answer.",
    "My reasoning was independent of any suggestions.",
    "I did consider the hint when answering."
]

# Load existing conversations
with open('data/mmlu/hint_self_report/gemma-3-4b-test/sycophancy/extended_conversations.jsonl', 'r') as f:
    conversations = [json.loads(line) for line in f]

# Update with test responses
for i, conv in enumerate(conversations):
    conv['messages'][-1]['content'] = test_responses[i]

# Save updated conversations
with open('data/mmlu/hint_self_report/gemma-3-4b-test/sycophancy/extended_conversations.jsonl', 'w') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')

print(f"Updated {len(conversations)} conversations with test responses")
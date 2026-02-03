"""View thought signatures from agent thinking logs."""

import json
from pathlib import Path
from datetime import datetime

def view_thought_signatures(limit=20):
    """Display recent thought signatures."""
    log_file = Path("logs/thought_signatures.jsonl")
    
    if not log_file.exists():
        print("No thought signatures logged yet.")
        print("Run the system and process some frames to generate thought signatures.")
        return
    
    signatures = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    sig = json.loads(line)
                    signatures.append(sig)
                except json.JSONDecodeError:
                    continue
    
    if not signatures:
        print("No thought signatures found in log file.")
        return
    
    print(f"\n{'='*80}")
    print(f"THOUGHT SIGNATURES LOG ({len(signatures)} total)")
    print(f"{'='*80}\n")
    
    for i, sig in enumerate(signatures[-limit:], 1):
        timestamp = datetime.fromtimestamp(sig.get("timestamp", 0))
        model = sig.get("model_version", "unknown")
        level = sig.get("thinking_level", "unknown")
        sig_id = sig.get("signature_id", "unknown")[:8]
        
        print(f"[{i}] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Model: {model} | Level: {level} | ID: {sig_id}")
        
        content = sig.get("content", {})
        if isinstance(content, dict):
            if "thought" in content:
                thought = str(content["thought"])[:200]
                print(f"    Thought: {thought}...")
            else:
                print(f"    Content: {str(content)[:200]}...")
        else:
            print(f"    Content: {str(content)[:200]}...")
        
        frame_ctx = sig.get("frame_context")
        if frame_ctx:
            frame_id = frame_ctx.get("frame_id", "?")
            print(f"    Frame: {frame_id}")
        
        print()
    
    print(f"{'='*80}")
    print(f"Showing last {min(limit, len(signatures))} of {len(signatures)} signatures")
    print(f"Full log: {log_file}")

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    view_thought_signatures(limit)

from datasets import load_dataset
import os

# -----------------------------
# Utility: save generic text
# -----------------------------
def save_dataset(dataset, path, num_samples=1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        count = 0

        for item in dataset:
            text = ""

            if isinstance(item, dict):
                if "text" in item and item["text"]:
                    text = item["text"]
                elif "content" in item and item["content"]:
                    text = item["content"]
                elif "dialog" in item:
                    text = " ".join(item["dialog"])

            text = text.strip().replace("\n", " ")

            if len(text) > 30:
                f.write(text + "\n")
                count += 1

            if count >= num_samples:
                break

    print(f"✅ Saved {count} samples → {path}", flush=True)


# -----------------------------
# Utility: save code dataset
# -----------------------------
def save_code_dataset(dataset, path, num_samples=1500):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        count = 0

        for item in dataset:
            # Correct field for CodeSearchNet
            text = item.get("func_code_string", "")

            text = text.strip().replace("\n", " ")

            if len(text) > 50:
                f.write(text + "\n")
                count += 1

            if count >= num_samples:
                break

    print(f"✅ Saved {count} code samples → {path}", flush=True)


print("🚀 Script started", flush=True)

# -----------------------------
# 1. GENERAL DOMAIN
# -----------------------------
print("📥 Loading WikiText...", flush=True)
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
save_dataset(wiki, "data/calibration/general/wiki.txt")


# -----------------------------
# 2. CODE DOMAIN
# -----------------------------
print("📥 Loading CodeSearchNet...", flush=True)
code = load_dataset("code_search_net", "python", split="train")
save_code_dataset(code, "data/calibration/code/code.txt")


# -----------------------------
# 3. CHAT DOMAIN
# -----------------------------
print("📥 Loading Chat dataset...", flush=True)
try:
    chat = load_dataset("OpenAssistant/oasst1", split="train")
except Exception as e:
    print("⚠️ OpenAssistant failed, using IMDB fallback...", flush=True)
    chat = load_dataset("imdb", split="train")

save_dataset(chat, "data/calibration/chat/chat.txt")


print("🎉 All datasets prepared successfully.", flush=True)
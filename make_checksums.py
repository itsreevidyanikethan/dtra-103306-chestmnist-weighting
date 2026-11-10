"""Write SHA-256 checksums of every artefact in outputs/ to CHECKSUMS.txt."""
import hashlib, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
lines = []
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        p = os.path.join(dirpath, name)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {p}")
open("CHECKSUMS.txt", "w").write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} checksums to CHECKSUMS.txt")

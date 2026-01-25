import fitz
import os

pdf_path = r"c:\Users\DELL\OneDrive - ku.lt\HORIZON_EUROPE\AI4WIND\DEPYTON\DEPONS-master\DEPONS 3.0 – TRACE 2023-05-30.pdf"
doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")
print("=" * 80)

# Read all pages and save to text file
all_text = []
for i, page in enumerate(doc):
    text = page.get_text()
    all_text.append(f"--- Page {i+1} ---\n{text}")
    
# Save to file
with open("DEPONS_TRACE.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

print("Text extracted to DEPONS_TRACE.txt")

# Print key sections
full_text = "\n".join(all_text)
print("\n=== KEY SECTIONS ===\n")

# Print first 20 pages
for i, page in enumerate(doc[:30]):
    print(f"--- Page {i+1} ---")
    print(page.get_text())
    print()

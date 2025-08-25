import os

root_dir = r"C:\Stand_Alone_Lidar_KPI_Evaluation"
output_file = r"C:\repo_dump.txt"

# Extensions you want to include
extensions = {
    ".cpp", ".h", ".py", ".txt", ".md", ".json", ".yaml", ".yml",
    ".cs", ".java", ".ini", ".cfg"
}

with open(output_file, "w", encoding="utf-8") as out:
    for foldername, subfolders, filenames in os.walk(root_dir):
        # Skip unwanted folders
        if ".git" in foldername or "__pycache__" in foldername:
            continue
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                filepath = os.path.join(foldername, filename)
                out.write(f"\n===== FILE: {filepath} =====\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        out.write(f.read())
                except UnicodeDecodeError:
                    out.write("[BINARY OR NON-TEXT FILE]\n")

print(f"✅ Dump complete! Saved to: {output_file}")

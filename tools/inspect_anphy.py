"""
Inspect the ANPHY dataset.

This script checks:

1. Folder structure
2. Subject folders
3. EDF files
4. Annotation files
5. Artifact files
6. Subject metadata
"""

from pathlib import Path


ROOT = Path("data/raw/ANPHY")


print("=" * 80)
print("ANPHY DATASET INSPECTOR")
print("=" * 80)
print()


# -----------------------------------------------------
# Root
# -----------------------------------------------------

print("Dataset Root")

print(ROOT)

print()

print("Exists :", ROOT.exists())

print()


# -----------------------------------------------------
# Subject Folder
# -----------------------------------------------------

subjects = ROOT / "Subjects"

print("Subjects Folder")

print(subjects)

print()

print("Exists :", subjects.exists())

print()

folders = sorted(

    [

        f

        for f in subjects.iterdir()

        if f.is_dir()

    ]

)

print("Number of subject folders :", len(folders))

print()


# -----------------------------------------------------
# Print all subjects
# -----------------------------------------------------

for folder in folders:

    print(folder.name)

print()


# -----------------------------------------------------
# Verify files
# -----------------------------------------------------

print("=" * 80)

print("VERIFY SUBJECTS")

print("=" * 80)

print()

errors = 0

for folder in folders:

    subject = folder.name

    edf = folder / f"{subject}.edf"

    annotation = folder / f"{subject}-annotation.txt"

    artifact = folder / f"{subject}_artndxn.mat"

    ok = True

    if not edf.exists():

        ok = False

    if not annotation.exists():

        ok = False

    if not artifact.exists():

        ok = False

    if ok:

        print(f"[PASS] {subject}")

    else:

        errors += 1

        print(f"[FAIL] {subject}")

print()

print("=" * 80)

print("SUMMARY")

print("=" * 80)

print()

print("Subjects :", len(folders))

print("Errors   :", errors)

print()


# -----------------------------------------------------
# Metadata
# -----------------------------------------------------

excel = ROOT / "Details information of subjects.xlsx"

print("Metadata Excel")

print(excel)

print()

print("Exists :", excel.exists())

print()

text = ROOT / "Co-registration average.txt"

print("Co-registration")

print(text)

print()

print("Exists :", text.exists())

print()

print("=" * 80)

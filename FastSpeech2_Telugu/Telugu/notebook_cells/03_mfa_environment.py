# Extracted from Telugu_FastSpeech2_Notebook.ipynb cell 41.
# Environment bootstrap for isolated MFA. Original notebook defaults are retained.

# 9D.1 Optional isolated MFA environment installation
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

MFA_ENV_PREFIX = Path("/content/mfa_telugu_env")
MFA_MAMBA_ROOT = Path("/content/mamba_root_mfa")
MICROMAMBA_BIN = Path(shutil.which("micromamba") or "/content/bin/micromamba")
MFA_TEMP_DIR = Path("/content/mfa_temp_telugu")
INSTALL_MFA_ENV_NOW = False  # Change to True only after the lexicon review.


def ensure_micromamba_binary():
    """Use Colab's existing micromamba or download its standalone Linux binary."""
    global MICROMAMBA_BIN
    found = shutil.which("micromamba")
    if found:
        MICROMAMBA_BIN = Path(found)
        return MICROMAMBA_BIN
    if MICROMAMBA_BIN.exists():
        return MICROMAMBA_BIN

    install_dir = Path("/content/bin")
    install_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path("/content/micromamba-linux-64.tar.bz2")
    url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
    print("Downloading micromamba bootstrap...")
    urllib.request.urlretrieve(url, archive_path)

    extract_dir = Path("/content/micromamba_extract")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = next((m for m in archive.getmembers() if m.name.endswith("bin/micromamba")), None)
        if member is None:
            raise RuntimeError("micromamba archive did not contain bin/micromamba")
        archive.extract(member, path=extract_dir)

    extracted = extract_dir / member.name
    destination = install_dir / "micromamba"
    shutil.copy2(extracted, destination)
    destination.chmod(0o755)
    MICROMAMBA_BIN = destination
    return MICROMAMBA_BIN


def mfa_environment_command(args, *, check=True):
    if not MICROMAMBA_BIN.exists() and not shutil.which("micromamba"):
        raise RuntimeError("micromamba is not installed. Run this cell with INSTALL_MFA_ENV_NOW=True.")

    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(MFA_MAMBA_ROOT)
    env["MFA_ROOT_DIR"] = str(Path("/content/mfa_root_telugu"))
    env["TMPDIR"] = str(MFA_TEMP_DIR)
    MFA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(MICROMAMBA_BIN), "run", "--prefix", str(MFA_ENV_PREFIX), *map(str, args)]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, env=env, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"MFA environment command failed with exit code {completed.returncode}")
    return completed


if INSTALL_MFA_ENV_NOW:
    if not MFA_TELUGU_LEXICON_REVIEWED:
        raise RuntimeError("Review the Telugu candidate lexicon first; leave INSTALL_MFA_ENV_NOW=False until then.")
    if not globals().get("LEXICON_ERRORS_RESOLVED", False):
        raise RuntimeError("Add manual pronunciations for all G2P-error words in telugu_lexicon_overrides.csv, rerun 8D.3, and then install MFA.")

    ensure_micromamba_binary()
    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = str(MFA_MAMBA_ROOT)
    if not MFA_ENV_PREFIX.exists():
        create_cmd = [
            str(MICROMAMBA_BIN), "create", "--yes",
            "--prefix", str(MFA_ENV_PREFIX),
            "--channel", "conda-forge",
            "montreal-forced-aligner",
        ]
        print("Creating isolated MFA environment. This may take several minutes...")
        subprocess.run(create_cmd, env=env, check=True)

    version_cmd = mfa_environment_command(["mfa", "--version"], check=True)
    print("MFA environment ready:", MFA_ENV_PREFIX)
else:
    print("MFA environment installation skipped (safe default).")
    print("After reviewing the lexicon, set INSTALL_MFA_ENV_NOW=True and rerun this cell.")

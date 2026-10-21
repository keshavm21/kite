"""A small structural check for kitec's PTX, used when `ptxas` is not installed.

It is NOT a substitute for ptxas (CI runs the real thing). It catches the mistakes ptxas reports most
often (docs/PTX_BACKEND.md section 10): undeclared or out-of-range registers, a register of the wrong class,
labels containing '.', branches to undefined labels, and missing header lines.
"""
import re

DECL = re.compile(r"\.reg\s+\.(pred|b32|f32|b64)\s+%(p|r|f|rd)<(\d+)>;")
REG = re.compile(r"%(rd|p|r|f)(\d+)\b")
SPECIAL = re.compile(r"%(tid|ntid|ctaid|nctaid)\.[xyz]")
LABEL_DEF = re.compile(r"^(\$[A-Za-z0-9_$]+):$")
BRA = re.compile(r"\bbra(?:\.uni)?\s+(\S+);")


def lint(text):
    """Returns a list of problems (empty if the file looks well-formed)."""
    problems = []
    for needle in (".version", ".target", ".address_size 64"):
        if needle not in text:
            problems.append(f"missing {needle}")
    for kernel in re.split(r"(?=^\.visible \.entry)", text, flags=re.M)[1:]:
        name = re.match(r"\.visible \.entry (\w+)", kernel).group(1)
        declared = {prefix: int(n) for _, prefix, n in DECL.findall(kernel)}
        labels, targets = set(), []
        for lineno, raw in enumerate(kernel.splitlines(), 1):
            line = raw.split("//")[0].strip()
            m = LABEL_DEF.match(line)
            if m:
                labels.add(m.group(1))
                continue
            if "$L_" in line and "." in line.split("$L_", 1)[1].split(";")[0]:
                problems.append(f"{name}: label with '.' in `{line}`")
            if line.startswith(".reg"):
                continue
            for b in BRA.findall(line):
                targets.append(b)
            for prefix, num in REG.findall(SPECIAL.sub("", line)):
                if prefix not in declared:
                    problems.append(f"{name}: %{prefix}{num} used but class not declared: `{line}`")
                elif int(num) >= declared[prefix]:
                    problems.append(f"{name}: %{prefix}{num} beyond .reg %{prefix}<{declared[prefix]}>: `{line}`")
        for t in targets:
            if t not in labels:
                problems.append(f"{name}: branch to undefined label {t}")
        if not re.search(r"^\s*ret;", kernel, flags=re.M):
            problems.append(f"{name}: no ret")
    return problems

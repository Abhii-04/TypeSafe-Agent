from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = (PROJECT_ROOT / "skills").resolve()

@tool
def Read_skill(skill:str)->str:
    """Read task related skills using this tool. """
    skill_path = (SKILLS_ROOT / skill /"SKILL.md").resolve()
   
    if skill_path not in PROJECT_ROOT:
        raise ValueError("Path must be inside project root.")
    
    if skill_path not in SKILLS_ROOT:
        raise  ValueError("skill path outside skill folder.")

    if not skill_path.is_file:
        return "File not found"
    content = skill_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            f"Skill '{skill}' has an empty SKILL.md. Add concrete instructions "
            f"to {skill_path.relative_to(PROJECT_ROOT)} before using this skill."
        )
    return content
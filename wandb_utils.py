import subprocess

def wandb_context(env_id: str,
                  algo: str,
                  variant: str,
                  seed: int,
                  hormone_enabled: bool,
                  project: str = "hormonal-rl",
                  extra_tags=None):
    """
    Returns kwargs for wandb.init with consistent naming:
      - name:   "{env}/{algo}/{variant}/seed{seed}-{gitsha}"
      - group:  "{env}-{algo}-{variant}"
      - tags:   ["env:*","algo:*","variant:*","seed:*","hormone:on|off", ...]
    """
    def _gitsha():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "nogit"

    git = _gitsha()
    name  = f"{env_id}/{algo}/{variant}/seed{seed}-{git}"
    group = f"{env_id}-{algo}-{variant}"
    tags  = [f"env:{env_id}", f"algo:{algo}", f"variant:{variant}",
             f"seed:{seed}", f"hormone:{'on' if hormone_enabled else 'off'}"]
    if extra_tags:
        tags.extend(list(extra_tags))

    return {
        "project": project,
        "name": name,
        "group": group,
        "tags": tags,
    }

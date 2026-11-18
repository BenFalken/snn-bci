# Source this from any shell or sbatch script to get the repository's paths as variables:
#     source "<repo>/snnbci_env.sh"
# It exports SNNBCI_ROOT, SNNBCI_WORKSPACE, SNNBCI_OSCAR_SCRATCH, SNNBCI_CORE, ... from paths.yaml (and
# paths.local.yaml, which overrides it: on Oscar, set workspace/oscar_scratch there). Needs python3 on the PATH.
_snnbci_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
eval "$(python3 "$_snnbci_here/paths.py" --sh)"
unset _snnbci_here

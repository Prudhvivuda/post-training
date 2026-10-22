#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SERVER="https://api.ai-dev06.kni.syseng.devcluster.openshift.com:6443"
PROJECT_NAME="team-b-lora"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="smoke"
PROFILE_EXPLICIT=false
RESUME_RUN_ID=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit-batch-job.sh [--profile smoke|contention]
  scripts/submit-batch-job.sh --resume RUN_ID

Each new run gets a dedicated PVC, source ConfigMap, and run ID. Resubmit a
preempted run with --resume RUN_ID to use its PVC and latest full checkpoint.
USAGE
}

while (($#)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PROFILE="$2"
      PROFILE_EXPLICIT=true
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RESUME_RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$PROFILE" != "smoke" && "$PROFILE" != "contention" ]]; then
  printf 'Invalid profile %s; choose smoke or contention.\n' "$PROFILE" >&2
  exit 2
fi

CURRENT_SERVER="$(oc whoami --show-server)"
if [[ "${CURRENT_SERVER%/}" != "${EXPECTED_SERVER%/}" ]]; then
  printf 'Refusing to submit the workload: current OpenShift server is %s; expected %s.\n' \
    "$CURRENT_SERVER" "$EXPECTED_SERVER" >&2
  exit 1
fi

if ! oc get namespace "$PROJECT_NAME" >/dev/null 2>&1; then
  printf 'OpenShift project %s does not exist; run scripts/create-project.sh first.\n' \
    "$PROJECT_NAME" >&2
  exit 1
fi

FS_GROUP_RANGE="$(oc get namespace "$PROJECT_NAME" \
  -o go-template='{{index .metadata.annotations "openshift.io/sa.scc.supplemental-groups"}}')"
if [[ -z "$FS_GROUP_RANGE" ]]; then
  FS_GROUP_RANGE="$(oc get namespace "$PROJECT_NAME" \
    -o go-template='{{index .metadata.annotations "openshift.io/sa.scc.uid-range"}}')"
fi
FS_GROUP_BLOCK="${FS_GROUP_RANGE%%,*}"
if [[ ! "$FS_GROUP_BLOCK" =~ ^([0-9]+)(/[0-9]+|-[0-9]+)?$ ]]; then
  printf 'Could not read a valid OpenShift supplemental group range from project %s.\n' \
    "$PROJECT_NAME" >&2
  exit 1
fi
FS_GROUP="${BASH_REMATCH[1]}"

if [[ -n "$RESUME_RUN_ID" ]]; then
  RUN_ID="$RESUME_RUN_ID"
else
  RUN_ID="run-$(date -u +%Y%m%dt%H%M%Sz)-${RANDOM}"
fi

if [[ ! "$RUN_ID" =~ ^run-[a-z0-9][a-z0-9-]{0,40}$ ]]; then
  printf 'Invalid run ID %s; use run- followed by lowercase letters, digits, and hyphens (up to 45 characters).\n' "$RUN_ID" >&2
  exit 2
fi

RUN_SUFFIX="${RUN_ID#run-}"
WORKSPACE_PVC="team-b-lora-workspace-${RUN_SUFFIX}"
SOURCE_CONFIGMAP="team-b-lora-source-${RUN_SUFFIX}"
RUN_LABEL="app.kubernetes.io/run-id=${RUN_ID}"

ACTIVE_JOBS="$(oc get jobs -n "$PROJECT_NAME" -l "$RUN_LABEL" \
  -o jsonpath='{range .items[?(@.status.active>0)]}{.metadata.name}{" "}{end}')"
if [[ -n "$ACTIVE_JOBS" ]]; then
  printf 'A Job for run %s is already active: %s\n' "$RUN_ID" "$ACTIVE_JOBS" >&2
  exit 1
fi
COMPLETED_JOBS="$(oc get jobs -n "$PROJECT_NAME" -l "$RUN_LABEL" \
  -o jsonpath='{range .items[?(@.status.succeeded>0)]}{.metadata.name}{" "}{end}')"
if [[ -n "$RESUME_RUN_ID" && -n "$COMPLETED_JOBS" ]]; then
  printf 'Run %s already has a completed Job: %s\n' "$RUN_ID" "$COMPLETED_JOBS" >&2
  exit 1
fi

if [[ -n "$RESUME_RUN_ID" ]]; then
  if ! oc get pvc "$WORKSPACE_PVC" -n "$PROJECT_NAME" >/dev/null 2>&1; then
    printf 'Cannot resume %s: workspace PVC %s was not found.\n' "$RUN_ID" "$WORKSPACE_PVC" >&2
    exit 1
  fi
  if ! oc get configmap "$SOURCE_CONFIGMAP" -n "$PROJECT_NAME" >/dev/null 2>&1; then
    printf 'Cannot resume %s: source ConfigMap %s was not found.\n' "$RUN_ID" "$SOURCE_CONFIGMAP" >&2
    exit 1
  fi
  STORED_PROFILE="$(oc get configmap "$SOURCE_CONFIGMAP" -n "$PROJECT_NAME" \
    -o jsonpath='{.data.profile}')"
  GIT_COMMIT="$(oc get configmap "$SOURCE_CONFIGMAP" -n "$PROJECT_NAME" \
    -o jsonpath='{.data.git_commit}')"
  GIT_WORKTREE_DIRTY="$(oc get configmap "$SOURCE_CONFIGMAP" -n "$PROJECT_NAME" \
    -o jsonpath='{.data.git_worktree_dirty}')"
  if [[ "$STORED_PROFILE" != "smoke" && "$STORED_PROFILE" != "contention" ]]; then
    printf 'Run %s has no valid saved profile in ConfigMap %s.\n' "$RUN_ID" "$SOURCE_CONFIGMAP" >&2
    exit 1
  fi
  if [[ "$PROFILE_EXPLICIT" == "true" && "$PROFILE" != "$STORED_PROFILE" ]]; then
    printf 'Run %s used profile %s; resume it with the same profile.\n' \
      "$RUN_ID" "$STORED_PROFILE" >&2
    exit 1
  fi
  PROFILE="$STORED_PROFILE"
else
  if oc get pvc "$WORKSPACE_PVC" -n "$PROJECT_NAME" >/dev/null 2>&1 || \
     oc get configmap "$SOURCE_CONFIGMAP" -n "$PROJECT_NAME" >/dev/null 2>&1; then
    printf 'Run ID collision: resources for %s already exist. Generate a new run.\n' "$RUN_ID" >&2
    exit 1
  fi
  GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ ! "$GIT_COMMIT" =~ ^[a-zA-Z0-9-]+$ ]]; then
    GIT_COMMIT="unknown"
  fi
  GIT_STATUS="$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null || true)"
  if [[ "$GIT_COMMIT" == "unknown" ]]; then
    GIT_WORKTREE_DIRTY="unknown"
  elif [[ -n "$GIT_STATUS" ]]; then
    GIT_WORKTREE_DIRTY="true"
  else
    GIT_WORKTREE_DIRTY="false"
  fi
  oc create configmap "$SOURCE_CONFIGMAP" \
    --from-file=requirements.txt="$ROOT_DIR/requirements.txt" \
    --from-file=prepare_data.py="$ROOT_DIR/data/prepare_data.py" \
    --from-file=run_workflow.py="$ROOT_DIR/scripts/run_workflow.py" \
    --from-literal=profile="$PROFILE" \
    --from-literal=git_commit="$GIT_COMMIT" \
    --from-literal=git_worktree_dirty="$GIT_WORKTREE_DIRTY" \
    --dry-run=client -o yaml | oc apply -n "$PROJECT_NAME" -f -
fi

sed \
  -e "s/__WORKSPACE_PVC__/${WORKSPACE_PVC}/g" \
  -e "s/__RUN_ID__/${RUN_ID}/g" \
  "$ROOT_DIR/deploy/batch-workspace-pvc.yaml" | oc apply -f -

JOB_NAME="$(sed \
  -e "s/__RUN_ID__/${RUN_ID}/g" \
  -e "s/__PROFILE__/${PROFILE}/g" \
  -e "s/__GIT_COMMIT__/${GIT_COMMIT}/g" \
  -e "s/__GIT_WORKTREE_DIRTY__/${GIT_WORKTREE_DIRTY}/g" \
  -e "s/__WORKSPACE_PVC__/${WORKSPACE_PVC}/g" \
  -e "s/__FS_GROUP__/${FS_GROUP}/g" \
  -e "s/__SOURCE_CONFIGMAP__/${SOURCE_CONFIGMAP}/g" \
  "$ROOT_DIR/deploy/batch-training-job.yaml" | oc create -n "$PROJECT_NAME" -f - \
    -o jsonpath='{.metadata.name}')"

printf 'Created Job %s for run %s in project %s.\n' "$JOB_NAME" "$RUN_ID" "$PROJECT_NAME"
printf 'Profile: %s; workspace PVC: %s.\n' "$PROFILE" "$WORKSPACE_PVC"
printf 'Scheduler: runai-scheduler; priority: low; preemptible: true; fsGroup: %s.\n' "$FS_GROUP"
printf 'Follow logs: oc logs -f job/%s -n %s\n' "$JOB_NAME" "$PROJECT_NAME"
printf 'After completion or preemption, collect scheduling evidence with:\n'
printf '  bash scripts/collect-batch-metrics.sh %s\n' "$RUN_ID"
if [[ -z "$RESUME_RUN_ID" ]]; then
  printf 'If preempted, resume this run with: bash scripts/submit-batch-job.sh --resume %s\n' "$RUN_ID"
fi

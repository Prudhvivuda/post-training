#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SERVER="https://api.ai-dev06.kni.syseng.devcluster.openshift.com:6443"
PROJECT_NAME="team-b-lora"
SOURCE_CONFIGMAP="team-b-lora-batch-source"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

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

ACTIVE_JOBS="$(oc get jobs -n "$PROJECT_NAME" -l app.kubernetes.io/component=batch-training \
  -o jsonpath='{range .items[?(@.status.active>0)]}{.metadata.name}{" "}{end}')"
if [[ -n "$ACTIVE_JOBS" ]]; then
  printf 'A Team B batch job is already active: %s\n' "$ACTIVE_JOBS" >&2
  exit 1
fi

oc create configmap "$SOURCE_CONFIGMAP" \
  --from-file=requirements.txt="$ROOT_DIR/requirements.txt" \
  --from-file=prepare_data.py="$ROOT_DIR/data/prepare_data.py" \
  --from-file=run_workflow.py="$ROOT_DIR/scripts/run_workflow.py" \
  --dry-run=client -o yaml | oc apply -n "$PROJECT_NAME" -f -
oc apply -f "$ROOT_DIR/deploy/batch-workspace-pvc.yaml"

JOB_NAME="$(oc create -f "$ROOT_DIR/deploy/batch-training-job.yaml" \
  -o jsonpath='{.metadata.name}')"
printf 'Created %s in project %s.\n' "$JOB_NAME" "$PROJECT_NAME"
printf 'Follow progress: oc logs -f job/%s -n %s\n' "$JOB_NAME" "$PROJECT_NAME"
printf 'Persistent outputs: PVC team-b-lora-batch-workspace at /workspace/project/outputs\n'

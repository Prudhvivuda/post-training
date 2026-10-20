#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SERVER="https://api.ai-dev06.kni.syseng.devcluster.openshift.com:6443"
PROJECT_NAME="${PROJECT_NAME:-team-b-lora}"
PROJECT_DISPLAY_NAME="Team B LoRA"
PROJECT_DESCRIPTION="Team B low-SLA LoRA fine-tuning workflow for RHOAI and Run:ai exploration."

CURRENT_SERVER="$(oc whoami --show-server)"
if [[ "${CURRENT_SERVER%/}" != "${EXPECTED_SERVER%/}" ]]; then
  printf 'Refusing to create the project: current OpenShift server is %s; expected %s.\n' \
    "$CURRENT_SERVER" "$EXPECTED_SERVER" >&2
  exit 1
fi

if oc get project "$PROJECT_NAME" >/dev/null 2>&1; then
  printf 'OpenShift project %s already exists; reusing it.\n' "$PROJECT_NAME"
else
  oc new-project "$PROJECT_NAME" \
    --display-name="$PROJECT_DISPLAY_NAME" \
    --description="$PROJECT_DESCRIPTION"
fi

# Opt the namespace into the OpenShift AI dashboard's data science project view.
oc label namespace "$PROJECT_NAME" opendatahub.io/dashboard=true --overwrite
oc project "$PROJECT_NAME"

printf 'Team B project is ready: %s\n' "$PROJECT_NAME"
printf 'Next: follow README.md to submit the Team B batch Job or create the optional interactive Workbench.\n'

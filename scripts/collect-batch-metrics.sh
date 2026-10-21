#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s RUN_ID\n' "$0" >&2
  exit 2
fi

RUN_ID="$1"
if [[ ! "$RUN_ID" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]]; then
  printf 'Invalid run ID %s.\n' "$RUN_ID" >&2
  exit 2
fi

PROJECT_NAME="team-b-lora"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/reports/cluster-runs"
REPORT_PATH="$REPORT_DIR/$RUN_ID.json"
RUN_LABEL="app.kubernetes.io/run-id=$RUN_ID"

JOB_JSON="$(oc get jobs -n "$PROJECT_NAME" -l "$RUN_LABEL" -o json)"
POD_JSON="$(oc get pods -n "$PROJECT_NAME" -l "$RUN_LABEL" -o json)"
CLUSTER_SERVER="$(oc whoami --show-server)"
export RUN_ID REPORT_PATH JOB_JSON POD_JSON CLUSTER_SERVER
mkdir -p "$REPORT_DIR"

python3 - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_between(start, end):
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds(), 3)


jobs = json.loads(os.environ["JOB_JSON"]).get("items", [])
pods = json.loads(os.environ["POD_JSON"]).get("items", [])
if not jobs:
    raise SystemExit(f"No batch Job found for run {os.environ['RUN_ID']}.")

job_records = []
for job in sorted(jobs, key=lambda item: item["metadata"].get("creationTimestamp", "")):
    metadata = job.get("metadata", {})
    status = job.get("status", {})
    job_records.append(
        {
            "name": metadata.get("name"),
            "created_at": metadata.get("creationTimestamp"),
            "start_time": status.get("startTime"),
            "completion_time": status.get("completionTime"),
            "succeeded": status.get("succeeded", 0),
            "failed": status.get("failed", 0),
            "created_to_started_seconds": seconds_between(
                metadata.get("creationTimestamp"), status.get("startTime")
            ),
            "created_to_completed_seconds": seconds_between(
                metadata.get("creationTimestamp"), status.get("completionTime")
            ),
        }
    )

pod_records = []
for pod in sorted(pods, key=lambda item: item["metadata"].get("creationTimestamp", "")):
    metadata = pod.get("metadata", {})
    spec = pod.get("spec", {})
    status = pod.get("status", {})
    scheduled_at = next(
        (
            condition.get("lastTransitionTime")
            for condition in status.get("conditions", [])
            if condition.get("type") == "PodScheduled" and condition.get("status") == "True"
        ),
        None,
    )
    container_statuses = status.get("containerStatuses", [])
    container_state = container_statuses[0].get("state", {}) if container_statuses else {}
    terminated = container_state.get("terminated", {})
    running = container_state.get("running", {})
    started_at = terminated.get("startedAt") or running.get("startedAt")
    finished_at = terminated.get("finishedAt")
    pod_created = metadata.get("creationTimestamp")
    pod_records.append(
        {
            "name": metadata.get("name"),
            "phase": status.get("phase"),
            "created_at": pod_created,
            "scheduled_at": scheduled_at,
            "container_started_at": started_at,
            "container_finished_at": finished_at,
            "node": spec.get("nodeName"),
            "scheduler": spec.get("schedulerName"),
            "termination_reason": terminated.get("reason"),
            "termination_exit_code": terminated.get("exitCode"),
            "pod_created_to_scheduled_seconds": seconds_between(pod_created, scheduled_at),
            "scheduled_to_container_started_seconds": seconds_between(scheduled_at, started_at),
            "pod_created_to_container_started_seconds": seconds_between(pod_created, started_at),
            "container_resources": [
                {
                    "name": container.get("name"),
                    "image": container.get("image"),
                    "requests": container.get("resources", {}).get("requests", {}),
                    "limits": container.get("resources", {}).get("limits", {}),
                }
                for container in spec.get("containers", [])
            ],
        }
    )

report = {
    "run_id": os.environ["RUN_ID"],
    "collected_at": datetime.now().astimezone().isoformat(),
    "cluster_server": os.environ["CLUSTER_SERVER"],
    "jobs": job_records,
    "pods": pod_records,
}
output = Path(os.environ["REPORT_PATH"])
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"Saved scheduling evidence to {output}")
PY

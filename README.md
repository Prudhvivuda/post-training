# Team B LoRA post-training demo

This project is the Team B RHOAI LoRA workflow for Jira NVIDIA-1537. It fine-tunes a small instruction model with LoRA on a deterministic, synthetic text-to-SQL dataset, saves the adapter, and compares base-model and adapted-model exact-match results on held-out prompts. The project bootstrap and workload launch are repeatable, and the batch Job is configured to use the Run:ai scheduler in the mapped project.

The project is designed to run from a GPU-enabled RHOAI Workbench or a batch Job in the Team B Data Science Project on `dev06`. The bootstrap creates or reuses only the `team-b-lora` OpenShift project and applies the OpenShift AI dashboard project label. Run:ai is installed and managed at the cluster level; an administrator separately maps the existing OpenShift namespace to a Run:ai project and configures its quota. Do not add Run:ai-managed labels or annotations in the bootstrap script.

## Workflow

1. Generate train and evaluation JSONL files from `data/prepare_data.py`.
2. Evaluate the base model on the held-out questions.
3. Fine-tune with Red Hat Training Hub `lora_sft`.
4. Evaluate the returned adapter model on the same questions.
5. Save run configuration, resumable Trainer checkpoints, the adapter, and `metrics.json` under the run directory.

The `smoke` profile uses the original 24-example training set for 2 epochs. The `contention` profile repeats those examples 8 times and trains for 5 epochs, producing a longer GPU workload for scheduling experiments. Both save full Trainer checkpoints every 10 steps by default.

The dataset uses fictional `employees` and `projects` tables. It is a small workflow and scheduling demo, not a meaningful SQL benchmark. Replace or expand it with a reviewed domain dataset after the RHOAI workflow is proven.

## RHOAI target

The default model is `Qwen/Qwen2.5-1.5B-Instruct`, using one GPU. The workflow pins Training Hub to `0.9.2`, the version recorded in the verified RHOAI 3.5.1 run, so the checkpoint bridge and runtime stay aligned with that image. The Training Hub support matrix includes Qwen 2.5 architectures for LoRA SFT, and its quick example uses this model family. RHOAI 3.5's customization guide describes Training Hub LoRA/QLoRA and the `Training | Jupyter | PyTorch | CUDA | Python` Workbench image. Check the installed RHOAI version, enabled components, image availability, and Team B quota on `dev06` before creating the Workbench.

References:

- [RHOAI 3.5: create a project](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/getting_started_with_red_hat_openshift_ai_self-managed/creating-a-project_get-started)
- [RHOAI 3.5 model customization and Training Hub](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/customize_models_for_gen_ai_and_agentic_ai_applications/train-the-model-by-using-your-prepared-data_custom-models)
- [Training Hub LoRA API and examples](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

## Deploy to `dev06`

Deploy this first version inside the **Team B RHOAI Data Science Project**. The expected namespace is `team-b-lora`. The bootstrap is scoped to the `dev06` API endpoint and refuses to run if your current `oc` context points elsewhere.

1. Log in to `dev06` with your approved OpenShift credentials. If you need a fresh login command, get it from the OpenShift console's **Copy login command** flow; do not save the token in this project. Confirm the active server:

   ```bash
   oc whoami --show-server
   ```

2. From this folder, create or reuse the Team B project:

   ```bash
   cd /Users/pvuda/Code/post-training/team-b-lora
   bash scripts/create-project.sh
   ```

   The script checks that `oc` is connected to `dev06`, creates `team-b-lora` if needed, opts it into the OpenShift AI dashboard, and selects it as the current project. If your account cannot create projects, ask the cluster/project owner to create `team-b-lora`; then rerun the script to apply the dashboard label.

3. In Run:ai, create a project named `team-b-lora` in the `default` department and associate it with the existing Kubernetes namespace `team-b-lora`. For the current low-SLA Team B profile, the project has 0 guaranteed GPUs, over-quota access enabled, and no maximum GPU cap; this lets training use spare GPUs while leaving its resources reclaimable. The project mapping adds namespace label `runai/queue=team-b-lora` and annotation `runai/enforce-scheduler-name=true`. These are managed by Run:ai and should not be added by `create-project.sh`.

4. Create the RHOAI 3.5.1 PyTorch CUDA Workbench and its persistent volume from the version-matched manifest:

   ```bash
   oc apply -f deploy/workbench.yaml
   oc get notebooks,pvc,pods -n team-b-lora -o wide
   ```

   The manifest requests one NVIDIA GPU, 2 CPUs, 8 GiB RAM, and a 20 GiB `gp3-csi` volume. It pins the currently published RHOAI 3.5 PyTorch image digest and tolerates dev06's `g5-gpu`/`g6e-gpu` worker taints. If the image, profile, or GPU-node taints change after a cluster upgrade, refresh this manifest from the installed ImageStream and node configuration before reapplying.

5. Put this directory in the Workbench's persistent home. Recommended: push it to an internal Git remote and clone it from JupyterLab. For a one-off transfer from your local machine, find the Workbench pod with `oc get pods -n team-b-lora` and run:

   ```bash
   oc exec -n team-b-lora <workbench-pod> -- mkdir -p /opt/app-root/src/team-b-lora
   oc rsync /Users/pvuda/Code/post-training/team-b-lora/ <workbench-pod>:/opt/app-root/src/team-b-lora/ -n team-b-lora
   ```

6. In the Workbench terminal, run:

   ```bash
   cd /opt/app-root/src/team-b-lora
   python -c 'from training_hub import lora_sft; print("Training Hub LoRA is available")'
   ```

   The RHOAI Training Workbench image may already include the required packages. If the import fails, first check the image/version with the RHOAI owner. If approved, install the LoRA extra and its Zoo dependency from the configured Python index:

   ```bash
   python -m pip install -r requirements.txt
   ```

7. Generate the data and launch the workflow. The output and Hugging Face cache default to this project directory, so keep the checkout under the Workbench's persistent home.

   ```bash
   python data/prepare_data.py
   python scripts/run_workflow.py
   ```

   Optional overrides:

   ```bash
   python scripts/run_workflow.py \
     --model-path Qwen/Qwen2.5-1.5B-Instruct \
     --num-epochs 2 \
     --output-dir /opt/app-root/src/team-b-lora/outputs/dev06-run-1
   ```

   If the workload exceeds the assigned GPU memory, retry with `--qlora` if the Workbench image and installed Training Hub LoRA extra support it.

8. Review `outputs/<run>/metrics.json` and the adapter/checkpoint directory. Record the Workbench image, RHOAI version, GPU type, Training Hub version, job duration, and any scheduling or resource errors for the Jira.

### Submit Team B as a batch workload

For the low-SLA batch use case, submit a Kubernetes Job from a local terminal after logging in to `dev06` and creating both project mappings. Each new run receives a unique PVC, source ConfigMap, and run ID, so concurrent jobs do not share their checkpoint workspace. The default `explicit` mode sets `schedulerName: runai-scheduler` plus Run:ai priority `low` and preemptibility `preemptible` labels. The launcher reads the namespace's OpenShift supplemental group range and sets the pod `fsGroup` so the dynamically provisioned PVC can be written without running as root:

```bash
cd /Users/pvuda/Code/post-training/team-b-lora
bash scripts/submit-batch-job.sh --profile smoke
oc get jobs,pods -n team-b-lora -l app.kubernetes.io/component=batch-training
oc logs -f job/<job-name> -n team-b-lora
```

For a longer contention workload, choose `contention`:

```bash
bash scripts/submit-batch-job.sh --profile contention
```

### Compare project defaults with workload-level settings

To check whether the Team B namespace/project mapping supplies Run:ai settings to a workload without explicit workload-level settings, submit the same smoke profile with `project-defaults` mode after the explicit-mode Job has completed:

```bash
bash scripts/submit-batch-job.sh --profile smoke --runai-mode project-defaults
```

This mode renders the same Job while omitting `schedulerName`, `priorityClassName`, and `kai.scheduler/preemptibility` from the Job and Pod template. It leaves the namespace mapping intact and records `runai_mode=project-defaults` in the run ConfigMap. Inspect the admitted Pod's `.spec.schedulerName` and Run:ai labels, then compare the workload in the Run:ai UI. A `runai-scheduler` value on the admitted Pod demonstrates scheduler selection from the integration path; absent priority/preemptibility labels mean the workload is using Run:ai's configured workload-type defaults unless the platform injects values. The evidence collector records the namespace mapping, effective Pod scheduler, workload labels, and submission mode:

```bash
bash scripts/collect-batch-metrics.sh <run-id>
```

Run the modes sequentially because each requests one GPU. Resume uses the mode saved with the original run, so a checkpointed run keeps the same workload-level or project-default configuration.

The launcher prints the run ID and PVC name. The Job snapshots the workflow source into a run-specific ConfigMap, creates a 20 GiB run-specific PVC, then installs requirements, creates deterministic data, evaluates the base model, runs LoRA SFT, and prints `metrics.json` to its logs. The run directory includes `run-config.json`, `run-state.json`, `base-evaluation.json`, checkpoints, and final metrics.

If the Job is preempted, Python runs as the container's main process and handles `SIGTERM` at the next optimizer step. It requests a full Trainer checkpoint and exits with code 75. `backoffLimit: 3` lets Kubernetes create a replacement pod up to three times; each replacement uses the same run-specific PVC and source snapshot, detects the existing run config, and resumes the latest full checkpoint. If an interruption happens before the first full checkpoint or the retry budget is exhausted, resubmit manually with the same run ID:

```bash
bash scripts/submit-batch-job.sh --resume <run-id>
```

The resume path checks that the saved model, Trainer state, and optimizer checkpoint exist and that the training configuration and dataset hashes match. It restores the latest checkpoint through Transformers' `resume_from_checkpoint` argument. If the preemption arrives before a full checkpoint is saved, the workflow reports that the run cannot be resumed; start a new run in that case.

Training Hub wraps the underlying TRL Trainer call, so the workflow installs a scoped runtime hook to attach the signal callback and forward the resume path. The hook restores the original methods after each training call and fails fast if the installed TRL Trainer lacks the resume argument.

After a run completes or is interrupted, collect OpenShift-side scheduling evidence (Job/Pod timestamps, scheduler, node, resource requests, and container image) to a local ignored report:

```bash
bash scripts/collect-batch-metrics.sh <run-id>
```

This writes `reports/cluster-runs/<run-id>.json`. The collector calculates pod creation-to-scheduling and scheduling-to-container-start intervals. It does not change cluster resources. The prior shared PVC `team-b-lora-batch-workspace` is left intact; new runs use run-specific PVCs.

### What this proves

The project is mapped to Run:ai on `dev06`; the default batch mode targets the Run:ai scheduler with explicit low priority and preemptibility, and `project-defaults` mode is available to inspect what the project integration supplies. The Oct 22 run verified Run:ai scheduling, a graceful signal-triggered checkpoint, and manual resume to completion on the same PVC. That validates checkpoint recovery for this workload; it does not prove scheduler-initiated cross-project reclaim. That still requires a coordinated workload in another Run:ai project. The Run:ai dashboard also displayed a cluster prerequisite warning while recognizing and allocating the Team B Job, so the cluster warning needs follow-up before treating the integration as production-ready.

## Verified `dev06` run (2026-10-19)

- OpenShift 4.22.13, RHOAI 3.5.1; namespace `team-b-lora` and a bound 20 GiB `gp3-csi` PVC.
- RHOAI PyTorch CUDA Workbench on an NVIDIA L40S, using the pinned image in `deploy/workbench.yaml`.
- Qwen2.5-1.5B-Instruct, Training Hub 0.9.2, 24 synthetic training examples, 6 held-out examples, 2 epochs, LoRA rank 8 / alpha 16.
- Training Hub reported 46.78 seconds; exact match was 1/6 for the base model and 3/6 after LoRA. This tiny synthetic set demonstrates end-to-end execution only; it is not a quality benchmark.
- Metrics are saved in `reports/dev06-run-20261019-metrics.json`. The adapter and checkpoint remain on the Workbench PVC at `/opt/app-root/src/team-b-lora/outputs/run-20261019T185710Z/adapter`.
- The Workbench used `default-scheduler`; the project had no ResourceQuota or LimitRange. Run:ai project mapping, policies, and quota behavior still need to be integrated and validated with the cluster team.
- The repeatable Kubernetes Job `team-b-lora-batch-8hzjh` completed on an NVIDIA A10G with the same base 1/6 and LoRA 3/6 exact-match results; Training Hub reported 56.86 seconds. Its metrics are saved in `reports/dev06-batch-run-20261019-metrics.json`, and the adapter/checkpoint persist on PVC `team-b-lora-batch-workspace` at `/workspace/project/outputs/run-20261019T190922Z/adapter`.
- Both workloads used `default-scheduler`. This validates the RHOAI batch execution path, not Run:ai project mapping, policy/quota inheritance, or scheduler behavior.

## Run:ai setup (2026-10-22)

- Run:ai dashboard version 3.0.20; cluster `ai-dev06` reported 5/5 GPU devices ready and no active Run:ai workloads before the Team B project was created.
- Created Run:ai project `team-b-lora` in department `default`, associated with the existing OpenShift namespace `team-b-lora`.
- Configured 0 guaranteed GPUs and enabled over-quota access with no maximum GPU cap. The project is opportunistic so its workloads can use free GPUs and can be reclaimed for another project's deserved quota. No scheduling or access rules were added.
- Verified the namespace has label `runai/queue=team-b-lora` and annotation `runai/enforce-scheduler-name=true`; the Run:ai project status is Ready.
- `deploy/batch-training-job.yaml` requests `runai-scheduler`, labels the Job and its Pod `low` / `preemptible`, and uses the OpenShift namespace-assigned group for writable run-specific PVCs. The container starts in `/workspace` so the non-root process creates its own project directory.
- The dashboard recognized the Team B Job and showed 1/1 GPU allocated, but also displayed “Prerequisites are missing from cluster ai-dev06.” The historical Oct 19 Workbench and batch runs used `default-scheduler`; neither is evidence for the new mapping or Run:ai scheduling behavior.
- In the Oct 22 verification, the RHOAI Notebook Workbench was running on `default-scheduler`; only the batch Job path was changed and verified with Run:ai. It was left running to preserve the active session. Recheck its current state before a demo, and include it in a scheduler comparison only after agreeing how an interactive workspace should be prioritized and reclaimed.

## Verified Run:ai checkpoint recovery (2026-10-22)

- Run `run-20261022t191831z-10294`, profile `contention`, ran on an NVIDIA A10G using `runai-scheduler`. Run:ai displayed the Job as Running with one requested and allocated GPU (24.15 GB reported memory).
- After step 764 of 960, a graceful SIGTERM was sent to the Team B pod to simulate an interruption. The handler saved `checkpoint-764` on PVC `team-b-lora-workspace-20261022t191831z-10294` and exited with code 75. That first Job (`team-b-lora-batch-d472p`) had `backoffLimit: 0`, so it became Failed rather than retrying automatically.
- `bash scripts/submit-batch-job.sh --resume run-20261022t191831z-10294` created `team-b-lora-batch-p457f` on the same PVC. The Trainer resumed from `checkpoint-764`, reached global step 960, wrote final metrics, and the Job completed. The resumed Training Hub call took 109.47 seconds; the completed Job took 194 seconds from creation to completion, including dependency installation and evaluation.
- The run's synthetic held-out exact-match score was 1/6 for the base model and 0/6 after LoRA. This is a scheduling/recovery run on a tiny synthetic dataset, not a model-quality result.
- OpenShift scheduling/timing evidence is in ignored `reports/cluster-runs/run-20261022t191831z-10294.json`. A competing workload was not started in another team's project, so scheduler-driven cross-project reclaim remains unverified.

## Verified Kubernetes retry (2026-10-22)

- Updated the batch template to `backoffLimit: 3`, so a failed or preempted pod can be replaced up to three times. Replacement pods reuse the same PVC and source ConfigMap; the workflow resumes the latest full Trainer checkpoint. Manual `--resume` remains available if retries are exhausted.
- In smoke run `run-20261022t192923z-13738`, deleting the first pod caused the Job controller to record one failed pod and create a replacement. The second pod reported `attempt: 2`, resumed from `checkpoint-48` on the same PVC, reached global step 48, wrote metrics, and the Job completed.
- This validates Job retry and checkpoint restoration after pod loss. The pod deletion was a controlled Team B test, not a Run:ai scheduler initiated cross-project reclaim. Its timing was near the end of the smoke run, so the checkpoint was already at the final training step.
- Scheduling evidence is in ignored `reports/cluster-runs/run-20261022t192923z-13738.json`.

## Local development

The training workflow requires a CUDA-enabled RHOAI Workbench and is not expected to run on a CPU-only laptop. Dataset generation is deterministic and can be run anywhere with Python 3:

```bash
python data/prepare_data.py
```


# Team B LoRA post-training demo

This project is the first RHOAI-only baseline for Jira NVIDIA-1537. It fine-tunes a small instruction model with LoRA on a deterministic, synthetic text-to-SQL dataset, saves the adapter, and compares base-model and adapted-model exact-match results on held-out prompts. The project bootstrap and workload launch are repeatable so Team B can recreate the project when the Run:ai project policy is settled.

The project is designed to run from a GPU-enabled RHOAI Workbench in the Team B Data Science Project on `dev06`. The bootstrap creates or reuses only the `team-b-lora` OpenShift project and applies the OpenShift AI dashboard project label. It does not install Run:ai, create cluster-wide resources, or add Run:ai scheduler, policy, or quota annotations. Add those after the team settles the integration mechanism, likely with Kyverno.

## Workflow

1. Generate train and evaluation JSONL files from `data/prepare_data.py`.
2. Evaluate the base model on the held-out questions.
3. Fine-tune with Red Hat Training Hub `lora_sft`.
4. Evaluate the returned adapter model on the same questions.
5. Save the adapter/checkpoints and `metrics.json` under the selected output directory.

The dataset uses fictional `employees` and `projects` tables. It is a small workflow and scheduling demo, not a meaningful SQL benchmark. Replace or expand it with a reviewed domain dataset after the RHOAI workflow is proven.

## RHOAI target

The default model is `Qwen/Qwen2.5-1.5B-Instruct`, using one GPU. The Training Hub support matrix includes Qwen 2.5 architectures for LoRA SFT, and its current quick example uses this model family. RHOAI 3.5's customization guide describes Training Hub LoRA/QLoRA and the `Training | Jupyter | PyTorch | CUDA | Python` Workbench image. Check the installed RHOAI version, enabled components, image availability, and Team B quota on `dev06` before creating the Workbench.

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

3. Create the RHOAI 3.5.1 PyTorch CUDA Workbench and its persistent volume from the version-matched manifest:

   ```bash
   oc apply -f deploy/workbench.yaml
   oc get notebooks,pvc,pods -n team-b-lora -o wide
   ```

   The manifest requests one NVIDIA GPU, 2 CPUs, 8 GiB RAM, and a 20 GiB `gp3-csi` volume. It pins the currently published RHOAI 3.5 PyTorch image digest and tolerates dev06's `g5-gpu`/`g6e-gpu` worker taints. If the image, profile, or GPU-node taints change after a cluster upgrade, refresh this manifest from the installed ImageStream and node configuration before reapplying.

4. Put this directory in the Workbench's persistent home. Recommended: push it to an internal Git remote and clone it from JupyterLab. For a one-off transfer from your local machine, find the Workbench pod with `oc get pods -n team-b-lora` and run:

   ```bash
   oc exec -n team-b-lora <workbench-pod> -- mkdir -p /opt/app-root/src/team-b-lora
   oc rsync /Users/pvuda/Code/post-training/team-b-lora/ <workbench-pod>:/opt/app-root/src/team-b-lora/ -n team-b-lora
   ```

5. In the Workbench terminal, run:

   ```bash
   cd /opt/app-root/src/team-b-lora
   python -c 'from training_hub import lora_sft; print("Training Hub LoRA is available")'
   ```

   The RHOAI Training Workbench image may already include the required packages. If the import fails, first check the image/version with the RHOAI owner. If approved, install the LoRA extra and its Zoo dependency from the configured Python index:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Generate the data and launch the workflow. The output and Hugging Face cache default to this project directory, so keep the checkout under the Workbench's persistent home.

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

7. Review `outputs/<run>/metrics.json` and the adapter/checkpoint directory. Record the Workbench image, RHOAI version, GPU type, Training Hub version, job duration, and any scheduling or resource errors for the Jira.

### Submit Team B as a batch workload

For the low-SLA batch use case, submit the repeatable Kubernetes Job from a local terminal after logging in to `dev06` and creating the project:

```bash
cd /Users/pvuda/Code/post-training/team-b-lora
bash scripts/submit-batch-job.sh
oc get jobs,pods -n team-b-lora -l app.kubernetes.io/component=batch-training
oc logs -f job/<job-name> -n team-b-lora
```

The launcher snapshots the workflow source into a ConfigMap, provisions a separate 20 GiB PVC, and creates a GPU Job using the RHOAI PyTorch image. The Job installs the requirements, generates its data, evaluates the base model, runs LoRA SFT, and prints the metrics JSON to its logs. The adapter, Hugging Face cache, and metrics persist on PVC `team-b-lora-batch-workspace` under `/workspace/project/outputs`. Submit another run after the active Job completes; the launcher prevents concurrent jobs from mounting the single-writer PVC.

### What this proves

This is the repeatable first RHOAI workload baseline, launched from a GPU Workbench. It does **not** yet prove Run:ai project annotation injection, quota borrowing/preemption, batch `Job`/Kubeflow Trainer scheduling, or concurrent Team A inference behavior. Once the cluster's Run:ai injection mechanism is agreed, the same script can be packaged as a batch workload for those tests.

## Verified `dev06` run (2026-10-19)

- OpenShift 4.22.13, RHOAI 3.5.1; namespace `team-b-lora` and a bound 20 GiB `gp3-csi` PVC.
- RHOAI PyTorch CUDA Workbench on an NVIDIA L40S, using the pinned image in `deploy/workbench.yaml`.
- Qwen2.5-1.5B-Instruct, Training Hub 0.9.2, 24 synthetic training examples, 6 held-out examples, 2 epochs, LoRA rank 8 / alpha 16.
- Training Hub reported 46.78 seconds; exact match was 1/6 for the base model and 3/6 after LoRA. This tiny synthetic set demonstrates end-to-end execution only; it is not a quality benchmark.
- Metrics are saved in `reports/dev06-run-20261019-metrics.json`. The adapter and checkpoint remain on the Workbench PVC at `/opt/app-root/src/team-b-lora/outputs/run-20261019T185710Z/adapter`.
- The Workbench used `default-scheduler`; the project had no ResourceQuota or LimitRange. Run:ai project mapping, policies, and quota behavior still need to be integrated and validated with the cluster team.
- The repeatable Kubernetes Job `team-b-lora-batch-8hzjh` completed on an NVIDIA A10G with the same base 1/6 and LoRA 3/6 exact-match results; Training Hub reported 56.86 seconds. Its metrics are saved in `reports/dev06-batch-run-20261019-metrics.json`, and the adapter/checkpoint persist on PVC `team-b-lora-batch-workspace` at `/workspace/project/outputs/run-20261019T190922Z/adapter`.
- Both workloads used `default-scheduler`. This validates the RHOAI batch execution path, not Run:ai project mapping, policy/quota inheritance, or scheduler behavior.

## Local development

The training workflow requires a CUDA-enabled RHOAI Workbench and is not expected to run on a CPU-only laptop. Dataset generation is deterministic and can be run anywhere with Python 3:

```bash
python data/prepare_data.py
```


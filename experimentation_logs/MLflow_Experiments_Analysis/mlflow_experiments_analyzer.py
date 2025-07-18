import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn

class MLflowExperimentAnalyzer:
    def __init__(self, experiment_name):
        self.client = MlflowClient()
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if self.experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        
        self.experiment_id = self.experiment.experiment_id
        print(f"[INFO] Using Experiment: {self.experiment.name} (ID: {self.experiment_id})")

    def list_top_runs(self, metric="best_val_pr_auc_cv", top_k=10):
        runs = self.client.search_runs(
            [self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_k
        )
        for run in runs:
            print(f"Run ID: {run.info.run_id}")
            print(f"  {metric}: {run.data.metrics.get(metric)}")
            for k, v in run.data.params.items():
                print(f"    {k}: {v}")
            print("-" * 40)
        return runs

    def load_best_model(self, metric="best_val_pr_auc_cv"):
        runs = self.client.search_runs(
            [self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        if not runs:
            print("No runs found.")
            return None
        best_run = runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"[INFO] Loading model from: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)

    def get_pruned_runs_by_drop(self, metric_base="final_pr_auc", drop_threshold=0.05):
        """
        Select runs where the drop from train to val on a given metric is within the threshold.

        For example, metric_base = "final_pr_auc" will compare:
            - "final_train_pr_auc"
            - "final_val_pr_auc"

        drop_threshold = 0.05 means at most 5% drop is allowed.
        """
        train_metric = f"{metric_base.replace('val', 'train')}"
        val_metric = f"{metric_base.replace('train', 'val')}"

        runs = self.client.search_runs([self.experiment_id])
        pruned_runs = []

        for run in runs:
            train_val = run.data.metrics.get(train_metric)
            val_val = run.data.metrics.get(val_metric)

            if train_val is None or val_val is None or train_val == 0:
                continue

            drop_ratio = (train_val - val_val) / train_val
            if drop_ratio <= drop_threshold:
                pruned_runs.append({
                    "run_id": run.info.run_id,
                    "train_score": train_val,
                    "val_score": val_val,
                    "drop_ratio": round(drop_ratio, 4),
                    "params": run.data.params
                })

        pruned_runs.sort(key=lambda x: x["val_score"], reverse=True)
        print(f"[INFO] Found {len(pruned_runs)} stable runs with ≤ {int(drop_threshold * 100)}% drop in {metric_base}")
        return pruned_runs


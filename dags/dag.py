from airflow import DAG
from airflow.models import TaskInstance
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator
from airflow.utils.session import provide_session
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import json
from pathlib import Path
import pickle


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_source_file(source_path):
    """Fail the dependency task if the source is missing or unreadable."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found or not a regular file: {path}")
    with path.open("rb") as source:
        source.read(1)
    print(f"Source file ready: {path}")


def check_store_partition(store_directory, partition_prefix, ds):
    """Require a completed Spark Parquet output for this run's snapshot."""
    partition = Path(store_directory) / f"{partition_prefix}{ds.replace('-', '_')}.parquet"
    check_source_file(partition / "_SUCCESS")
    parts = sorted(partition.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet part files found: {partition}")
    for part in parts:
        check_source_file(part)
    print(f"Store partition ready: {partition}")


def inference_period_ready(ds, **context):
    # July–December 2023 labels first support deployment in January 2024.
    return datetime.strptime(ds, "%Y-%m-%d") >= datetime(2024, 1, 1)


def performance_cohort_date(ds):
    snapshot = datetime.strptime(ds, "%Y-%m-%d")
    if snapshot.day != 1:
        raise ValueError("Snapshot date must be the first day of a month")
    return (snapshot - relativedelta(months=6)).strftime("%Y-%m-%d")


def performance_period_ready(ds, **context):
    return inference_period_ready(performance_cohort_date(ds))


def model_training_date(ds):
    snapshot = datetime.strptime(ds, "%Y-%m-%d")
    if snapshot.day != 1:
        raise ValueError("Snapshot date must be the first day of a month")
    if not inference_period_ready(ds):
        raise ValueError("No deployed model before 2024-01-01")
    return snapshot.replace(month=1 if snapshot.month <= 6 else 7)


def model_name_for_date(ds):
    return f"credit_model_{model_training_date(ds):%Y_%m_%d}.pkl"


def choose_training_path(ds, **context):
    if model_training_date(ds).strftime("%Y-%m-%d") == ds:
        return "train_model"
    return "reuse_existing_model"


@provide_session
def validate_model_ready(ds, dag_run, model_bank="/app/model_bank", session=None, **context):
    """Require successful training and its exact model/reference before scoring."""
    cutoff = model_training_date(ds)
    logical_date = cutoff.replace(tzinfo=timezone.utc)

    def training_task(dag_id):
        return session.query(TaskInstance).filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.task_id == "train_model",
            TaskInstance.execution_date == logical_date,
        ).one_or_none()

    training = training_task(dag_run.dag_id)
    if training is None:
        # Migration only: the retired DAG's completed runs remain valid proof.
        # Never fall back when an inline task exists but failed/is incomplete.
        training = training_task("model_training_dag")
        if training is not None and training.dag_run.conf.get("snapshotdate") != cutoff.strftime("%Y-%m-%d"):
            raise ValueError("Legacy training cutoff does not match the selected version")
    if training is None or training.state != "success":
        raise ValueError(f"Successful training is required for {cutoff:%Y-%m-%d}; replay that version's training run first")

    name = model_name_for_date(ds)
    bank = Path(model_bank)
    with (bank / name).open("rb") as file:
        artefact = pickle.load(file)  # Trusted local model in the pinned environment.
    if (artefact["model_version"] != Path(name).stem
            or artefact["data_dates"]["model_train_date"].date() != cutoff.date()):
        raise ValueError("Model artefact does not match the selected historical version")
    reference = json.loads((bank / f"{Path(name).stem}_psi_reference.json").read_text())
    metadata = reference["metadata"]
    if (metadata["model_name"] != name
            or metadata["deployment_date"] != cutoff.strftime("%Y-%m-%d")
            or datetime.fromisoformat(metadata["end_date"]).date() >= cutoff.date()):
        raise ValueError("PSI reference does not match the selected historical version")
    print(f"Model ready: {name}; successful training in {training.dag_id}/{training.run_id}")
    return name


with DAG(
    'data_processing_dag',
    default_args=default_args,
    description='Monthly processing, training, inference, PSI and delayed model performance',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=True,
    user_defined_macros={
        "model_name_for_date": model_name_for_date,
        "performance_cohort_date": performance_cohort_date,
    },
) as dag:

    def source_check(task_id, csv_file):
        return PythonOperator(
            task_id=task_id,
            python_callable=check_source_file,
            op_kwargs={"source_path": f"/app/data/{csv_file}"},
        )

    dep_check_source_label_data = source_check(
        "dep_check_source_label_data", "lms_loan_daily.csv")
    dep_check_source_data_bronze_1 = source_check(
        "dep_check_source_data_bronze_1", "features_attributes.csv")
    dep_check_source_data_bronze_2 = source_check(
        "dep_check_source_data_bronze_2", "features_financials.csv")
    dep_check_source_data_bronze_3 = source_check(
        "dep_check_source_data_bronze_3", "feature_clickstream.csv")

    def bronze_task(task_id, csv_file, bronze_subdir, prefix):
        return BashOperator(
            task_id=task_id,
            bash_command=(
                'cd /app && '
                'PYTHONPATH=/app python3 scripts/bronze_table.py '
                '--snapshotdate "{{ ds }}" '
                f'--csv_file_path "/app/data/{csv_file}" '
                f'--bronze_directory "/app/datamart/bronze/{bronze_subdir}" '
                f'--partition_prefix "{prefix}"'
            ),
        )

    bronze_label_store = bronze_task(
        "bronze_label_store", "lms_loan_daily.csv", "lms", "bronze_loan_daily_")
    bronze_customers_table = bronze_task(
        "bronze_customers_table", "features_attributes.csv", "customers", "bronze_customers_")
    bronze_financials_table = bronze_task(
        "bronze_financials_table", "features_financials.csv", "financials", "bronze_financials_")
    bronze_clickstream_table = bronze_task(
        "bronze_clickstream_table", "feature_clickstream.csv", "clickstream", "bronze_clickstream_")

    dep_check_source_label_data >> bronze_label_store
    dep_check_source_data_bronze_1 >> bronze_customers_table
    dep_check_source_data_bronze_2 >> bronze_financials_table
    dep_check_source_data_bronze_3 >> bronze_clickstream_table

    silver_label_store = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/silver_label_store.py '
            '--snapshotdate "{{ ds }}"'
            
        ),
    )

    silver_customers_table = BashOperator(
        task_id='silver_customers_table',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/silver_customers_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    silver_financials_table = BashOperator(
        task_id='silver_financials_table',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/silver_financials_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    silver_clickstream_table = BashOperator(
        task_id='silver_clickstream_table',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/silver_clickstream_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    bronze_label_store >> silver_label_store
    bronze_customers_table >> silver_customers_table
    bronze_financials_table >> silver_financials_table
    bronze_clickstream_table >> silver_clickstream_table

    gold_feature_store = BashOperator(
        task_id='gold_feature_store',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/gold_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 scripts/gold_label_store.py '
            '--snapshotdate "{{ ds }}" '
            '--dpd "30" '
            '--mob "6"'
        ),
    )

    silver_customers_table >> gold_feature_store
    silver_financials_table >> gold_feature_store
    silver_clickstream_table >> gold_feature_store
    silver_label_store >> gold_label_store

    feature_store_completed = PythonOperator(
        task_id="feature_store_completed",
        python_callable=check_store_partition,
        op_kwargs={
            "store_directory": "/app/datamart/gold/features_store",
            "partition_prefix": "gold_features_store_",
        },
    )
    label_store_completed = PythonOperator(
        task_id="label_store_completed",
        python_callable=check_store_partition,
        op_kwargs={
            "store_directory": "/app/datamart/gold/label_store",
            "partition_prefix": "gold_label_store_",
        },
    )
    gold_feature_store >> feature_store_completed
    gold_label_store >> label_store_completed


    # Warm-up dates build historical data without scoring with a future model.
    check_inference_period = ShortCircuitOperator(
        task_id="check_inference_period",
        python_callable=inference_period_ready,
    )

    check_retraining = BranchPythonOperator(
        task_id="check_retraining",
        python_callable=choose_training_path,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 -u scripts/model_train.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    reuse_existing_model = DummyOperator(task_id="reuse_existing_model")
    model_ready = PythonOperator(
        task_id="model_ready",
        python_callable=validate_model_ready,
        trigger_rule="none_failed_min_one_success",
    )

    # Every script uses the processing date directly. Both model tasks select
    # the same January/July version, never the newest file in model_bank.
    model_inference_start = BashOperator(
        task_id="model_inference_start",
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 -u scripts/model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--model "{{ model_name_for_date(ds) }}"'
        ),
    )

    model_psi_monitoring = BashOperator(
        task_id="model_psi_monitoring",
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 -u scripts/model_monitoring.py monitor '
            '--snapshotdate "{{ ds }}" '
            '--model "{{ model_name_for_date(ds) }}"'
        ),
    )

    # Independent of the current model: evaluate the cohort whose six-month
    # outcomes have just arrived, even if current training/inference fails.
    check_performance_period = ShortCircuitOperator(
        task_id="check_performance_period",
        python_callable=performance_period_ready,
    )

    model_performance_monitoring = BashOperator(
        task_id="model_performance_monitoring",
        bash_command=(
            'cd /app && '
            'PYTHONPATH=/app python3 -u scripts/model_performance.py '
            '--snapshotdate "{{ ds }}" '
            '--model "{{ model_name_for_date(performance_cohort_date(ds)) }}"'
        ),
    )

    label_store_completed >> check_performance_period >> model_performance_monitoring
    [feature_store_completed, label_store_completed] >> check_inference_period
    check_inference_period >> check_retraining
    check_retraining >> [train_model, reuse_existing_model]
    [train_model, reuse_existing_model] >> model_ready
    model_ready >> model_inference_start >> model_psi_monitoring
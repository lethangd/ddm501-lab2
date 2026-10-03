# =============================================================================
# Airflow image with the pipeline's dependencies
# DDM501 - Lab 2
#
# TODO: Complete this Dockerfile.
#
# =============================================================================
FROM apache/airflow:2.8.4-python3.11

# The base image starts as root; pip install as root writes to the wrong
# site-packages and the scheduler would not see the packages.
USER airflow

# Install with Airflow's constraints so our packages cannot upgrade or downgrade
# anything Airflow itself depends on.
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.11.txt"

# docker-compose.yml mounts ./pipeline to /opt/airflow/project/pipeline.
ENV PYTHONPATH="/opt/airflow/project"
